from numba import njit, void, int32, int64, float64, complex128, types

from numpy import (pi, sin, cos, exp, sqrt, log, arctan2,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array)
import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.phys_const import mu0, epsilon0
from petram.phys.numba_coefficient import NumbaCoefficient
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff
from petram.phys.vtable import VtableElement, Vtable

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

vtable_data0 = [('B', VtableElement('bext', type='array',
                                    guilabel='magnetic field',
                                    default="=[0.,0.,0.]",
                                    tip="external magnetic field")),
                ('dens_e', VtableElement('dens_e', type='float',
                                         guilabel='electron density(m-3)',
                                         default="1e19",
                                         tip="electron density")),
                ('temperature', VtableElement('temperature', type='float',
                                              guilabel='Tc(ev) or w_col(1/s)',
                                              default="10.",
                                              tip="angular collision freq. or effecitive temperature for collisions")),
                ('dens_i', VtableElement('dens_i', type='array',
                                         guilabel='ion densities(m-3)',
                                         default="0.9e19, 0.1e19",
                                         tip="ion densities")),
                ('mass', VtableElement('mass', type='array',
                                       guilabel='ion masses(/Da)',
                                       default="2, 1",
                                       no_func=True,
                                       tip="mass. normalized by atomic mass unit")),
                ('charge_q', VtableElement('charge_q', type='array',
                                           guilabel='ion charges(/q)',
                                           default="1, 1",
                                           no_func=True,
                                           tip="ion charges normalized by q(=1.60217662e-19 [C])")), ]

stix_options = ("SDP", "SD", "SP", "DP", "P", "w/o xx", "None")
default_stix_option = "(default) include all"

col_model_options = ["w/o col.", "Tc", "wcol",
                     "Tc (col. only)", "wcol (col. only)"]
default_col_model = col_model_options[1]


def value2int(num_ions, value):
    '''
    GUI value to interger
    '''
    if value == default_stix_option:
        return [0]*(num_ions+1)

    panelvalue = [x.split(":")[-1].strip() for x in value.split(",")]
    return [stix_options.index(x) for x in panelvalue]


def build_coefficients(ind_vars, omega, B, dens_e, t_e, dens_i, masses, charges, col_model,
                       g_ns, l_ns, sdim=3, terms=default_stix_option):

    from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold_std,
                                                                   epsilonr_pl_cold_g,
                                                                   epsilonr_pl_cold,
                                                                   epsilonr_pl_cold_generic,
                                                                   f_collisions)

    Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

    masses = np.array(masses, dtype=np.float64) * Da
    charges = np.array(charges, dtype=np.int32)

    num_ions = len(masses)
    l = l_ns
    g = g_ns

    B_coeff = VCoeff(3, [B], ind_vars, l, g,
                     return_complex=False, return_mfem_constant=True)
    dens_e_coeff = SCoeff([dens_e, ], ind_vars, l, g,
                          return_complex=False, return_mfem_constant=True)
    t_e_coeff = SCoeff([t_e, ], ind_vars, l, g,
                       return_complex=False, return_mfem_constant=True)
    dens_i_coeff = VCoeff(num_ions, [dens_i, ], ind_vars, l, g,
                          return_complex=False, return_mfem_constant=True)

    col_model = col_model_options.index(col_model)
    params = {'omega': omega, 'masses': masses, 'charges': charges,
              'col_model': col_model}

    if True:
        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            e_cold = epsilonr_pl_cold(
                omega, B, dens_i, masses, charges, t_e, dens_e, col_model)
            out = -epsilon0 * omega * omega*e_cold
            return out

        def sdp(ptx, B, dens_e, t_e, dens_i):
            out = epsilonr_pl_cold_std(
                omega, B, dens_i, masses, charges, t_e, dens_e, col_model)
            return out

    else:
        terms = value2int(len(charges), terms)
        terms = np.array(terms, dtype=np.int32)
        params["sterms"] = terms

        def epsilonr(ptx, B, dens_e, t_e, dens_i):
            out = -epsilon0 * omega * omega*epsilonr_pl_cold_generic(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms, col_model)
            return out

        def sdp(ptx, B, dens_e, t_e, dens_i):
            out = epsilonr_pl_cold_g(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms, col_model)
            return out

    if col_model > 2:
        def mur(ptx):
            return 10000.*mu0*np.eye(3, dtype=np.complex128)
    else:
        def mur(ptx):
            return mu0*np.eye(3, dtype=np.complex128)

    def sigma(ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    def nuei(ptx, dens_e, t_e, dens_i):
        # iidx : index of ions
        nuei = f_collisions(dens_i, charges, t_e, dens_e)
        return nuei[iidx]

    numba_debug = False if myid != 0 else get_numba_debug()

    dependency = (B_coeff, dens_e_coeff, t_e_coeff, dens_i_coeff)
    dependency = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                  for x in dependency]

    jitter = mfem.jit.matrix(sdim=sdim, shape=(3, 3), complex=True, params=params,
                             debug=numba_debug, dependency=dependency)
    mfem_coeff1 = jitter(epsilonr)

    jitter2 = mfem.jit.matrix(sdim=sdim, shape=(3, 3), complex=True, params=params,
                              debug=numba_debug)
    mfem_coeff2 = jitter2(mur)
    mfem_coeff3 = jitter2(sigma)
    mfem_coeff4 = jitter(sdp)

    coeff1 = NumbaCoefficient(mfem_coeff1)
    coeff2 = NumbaCoefficient(mfem_coeff2)
    coeff3 = NumbaCoefficient(mfem_coeff3)
    coeff4 = NumbaCoefficient(mfem_coeff4)

    dependency3 = (dens_e_coeff, t_e_coeff, dens_i_coeff)
    dependency3 = [(x.mfem_numba_coeff if isinstance(x, NumbaCoefficient) else x)
                   for x in dependency3]
    jitter3 = mfem.jit.scalar(sdim=sdim, complex=False, params=params, debug=numba_debug,
                              dependency=dependency3)
    coeff5 = []
    for idx in range(len(masses)):
        params['iidx'] = idx
        mfem_coeff5 = jitter3(nuei)
        coeff5.append(NumbaCoefficient(mfem_coeff5))

    return coeff1, coeff2, coeff3, coeff4, coeff5


def build_variables(solvar, ss, ind_vars, omega, B, dens_e, t_e, dens_i, masses, charges,
                    col_model, g_ns, l_ns, terms=default_stix_option):

    import petram.phys.common as pcomm
    # from petram.phys.common.rf_dispersion_coldplasma_numba import (epsilonr_pl_cold_std,
    #                                                               epsilonr_pl_cold_g,
    #                                                               epsilonr_pl_cold,
    #                                                               epsilonr_pl_cold_generic,
    #                                                               f_collisions)
    # from petram.phys.common.rf_plasma_wc_wp import wpesq, wpisq, wce, wci

    Da = 1.66053906660e-27      # atomic mass unit (u or Dalton) (kg)

    masses = np.array(masses, dtype=np.float64) * Da
    charges = np.array(charges, dtype=np.int32)

    num_ions = len(masses)
    l = l_ns
    g = g_ns

    def func(x, dens_smooth=None):
        return dens_smooth

    from petram.helper.variables import (variable,
                                         Constant,
                                         ExpressionVariable,
                                         NumbaCoefficientVariable,
                                         PyFunctionVariable)
    d1 = variable.jit.float(dependency=("dens_smooth",))(func)

    def make_variable(x):
        if isinstance(x, str):
            d1 = ExpressionVariable(x, ind_vars, gns=g_ns)
        else:
            d1 = Constant(x)
        return d1

    B_var = make_variable(B)
    te_var = make_variable(t_e)
    dense_var = make_variable(dens_e)
    densi_var = make_variable(dens_i)

    col_model = col_model_options.index(col_model)
    params = {'omega': omega, 'masses': masses,
              'charges': charges, 'col_model': col_model, 'pcomm': pcomm}

    if True:
        def epsilonr(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            from petram.phys.common.rf_dispersion_coldplasma_numba import epsilonr_pl_cold

            out = -epsilon0 * omega * omega*epsilonr_pl_cold(
                omega, B, dens_i, masses, charges, t_e, dens_e, col_model)
            return out

        def sdp(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            from petram.phys.common.rf_dispersion_coldplasma_numba import epsilonr_pl_cold_std
            out = epsilonr_pl_cold_std(
                omega, B, dens_i, masses, charges, t_e, dens_e, col_model)
            return out

        def epsilonrac(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            from petram.phys.common.rf_dispersion_coldplasma_numba import epsilonr_pl_cold
            out = -epsilon0 * omega * omega*epsilonr_pl_cold(
                omega, B, dens_i, masses, charges, t_e, dens_e, col_model)
            return (out - out.transpose().conj())/2.0

    else:
        terms = value2int(len(charges), terms)
        terms = np.array(terms, dtype=np.int32)
        params["sterms"] = terms

        def epsilonr(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            from petram.phys.common.rf_dispersion_coldplasma_numba import epsilonr_pl_cold_generic

            out = -epsilon0 * omega * omega*epsilonr_pl_cold_generic(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms, col_model)
            return out

        def sdp(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            from petram.phys.common.rf_dispersion_coldplasma_numba import epsilonr_pl_cold_g
            out = epsilonr_pl_cold_g(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms, col_model)
            return out

        def epsilonrac(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
            from petram.phys.common.rf_dispersion_coldplasma_numba import epsilonr_pl_cold_generic
            out = -epsilon0 * omega * omega*epsilonr_pl_cold_generic(
                omega, B, dens_i, masses, charges, t_e, dens_e, sterms, col_model)
            return (out - out.transpose().conj())/2.0

    if col_model > 2:
        def mur(*_ptx):
            return 100000.*mu0*np.eye(3, dtype=np.complex128)
    else:
        def mur(*_ptx):
            return mu0*np.eye(3, dtype=np.complex128)

    def mur(*_ptx):
        return mu0*np.eye(3, dtype=np.complex128)

    def sigma(*_ptx):
        return - 1j*omega * np.zeros((3, 3), dtype=np.complex128)

    def nuei(*_ptx, dens_e=dens_e, t_e=t_e, dens_i=dens_i):
        from petram.phys.common.rf_dispersion_coldplasma_numba import f_collisions

        nuei = f_collisions(dens_i, charges, t_e, dens_e)
        return nuei

    def fce(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
        from petram.phys.common.rf_plasma_wc_wp import wce

        freq = omega/2/pi
        b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
        fce = wce(b_norm, freq)*omega/2/pi
        return fce

    def fci(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
        from petram.phys.phys_const import Da
        from petram.phys.common.rf_plasma_wc_wp import wci

        freq = omega/2/pi
        fci = np.zeros(len(masses))
        b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)

        i = 0
        for mass, charge in zip(masses, charges):
            A = mass/Da
            Z = charge
            fci[i] = wci(b_norm, freq, A, Z)*omega/2/pi
            i = i+1

        return fci

    def fpe(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
        from petram.phys.common.rf_plasma_wc_wp import wpesq

        freq = omega/2/pi
        fpe = sqrt(wpesq(dens_e, freq)*omega**2)/2/pi
        return fpe

    def fpi(*_ptx, B=None, dens_e=None, t_e=None, dens_i=None):
        from petram.phys.phys_const import Da
        from petram.phys.common.rf_plasma_wc_wp import wpisq

        freq = omega/2/pi
        fpi = np.zeros(len(masses))

        i = 0
        for dens, mass, charge in zip(dens_i, masses, charges):
            A = mass/Da
            Z = charge
            fpi[i] = sqrt(wpisq(dens, A, Z, freq)*omega**2)/2/pi
            i = i+1

        return fpi

    solvar["_B_"+ss] = B_var
    solvar["_ne_"+ss] = dense_var
    solvar["_te_"+ss] = te_var
    solvar["_ni_"+ss] = densi_var
    dependency = ("_B_"+ss, "_ne_"+ss, "_te_"+ss, "_ni_"+ss)

    var1 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(epsilonr)
    var2 = variable.array(complex=True, shape=(3, 3),
                          params=params)(mur)
    var3 = variable.array(complex=True, shape=(3, 3),
                          params=params)(sigma)
    var4 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(sdp)
    var5 = variable.array(complex=True, shape=(len(masses),),
                          dependency=dependency, params=params)(nuei)
    var6 = variable.array(complex=True, shape=(3, 3),
                          dependency=dependency, params=params)(epsilonrac)

    var7 = variable.float(dependency=dependency, params=params)(fce)
    var8 = variable.array(complex=False, shape=(len(masses),),
                          dependency=dependency, params=params)(fci)

    var9 = variable.float(dependency=dependency, params=params)(fpe)
    var10 = variable.array(complex=False, shape=(len(masses),),
                           dependency=dependency, params=params)(fpi)

    return var1, var2, var3, var4, var5, var6, var7, var8, var9, var10


def add_domain_variables_common(obj, ret, v, suffix, ind_vars):
    ss = obj.parent.parent.name()+'_' + obj.name()  # phys module name + name

    v["_e_"+ss] = ret[0]
    v["_m_"+ss] = ret[1]
    v["_s_"+ss] = ret[2]
    v["_spd_"+ss] = ret[3]
    v["_nuei_"+ss] = ret[4]
    v["_eac_"+ss] = ret[5]
    v["_fce_"+ss] = ret[6]
    v["_fci_"+ss] = ret[7]
    v["_fpe_"+ss] = ret[8]
    v["_fpi_"+ss] = ret[9]

    obj.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', [
        "_e_"+ss + "/(-omega*omega*e0)"], ["omega"])
    obj.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonrac', [
        "_eac_"+ss + "/(-omega*omega*e0)"], ["omega"])

    obj.do_add_scalar_expr(v, suffix, ind_vars, "Pcol",
                           "omega*conj(E).dot(epsilonrac.dot(E))/1j*e0",
                           vars=['E', 'epsilonrac', 'omega'],)

    obj.do_add_scalar_expr(v, suffix, ind_vars, 'fce', "_fce_"+ss)
    obj.do_add_matrix_expr(v, suffix, ind_vars, 'fci', ["_fci_"+ss])
    obj.do_add_scalar_expr(v, suffix, ind_vars, 'fpe', "_fpe_"+ss)
    obj.do_add_matrix_expr(v, suffix, ind_vars, 'fpi', ["_fpi_"+ss])

    obj.do_add_matrix_expr(v, suffix, ind_vars,
                           'mur', ["_m_"+ss + "/mu0"])
    obj.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', [
        "_s_"+ss + "/(-1j*omega)"])
    obj.do_add_matrix_expr(v, suffix, ind_vars, 'nuei', ["_nuei_"+ss])
    obj.do_add_matrix_expr(v, suffix, ind_vars,
                           'Sstix', ["_spd_"+ss+"[0,0]"])
    obj.do_add_matrix_expr(v, suffix, ind_vars, 'Dstix', [
        "1j*_spd_"+ss+"[0,1]"])
    obj.do_add_matrix_expr(v, suffix, ind_vars,
                           'Pstix', ["_spd_"+ss+"[2,2]"])
    obj.do_add_matrix_expr(v, suffix, ind_vars,
                           'Rstix', ["_spd_"+ss+"[0,0] + 1j*_spd_"+ss+"[0,1]"])
    obj.do_add_matrix_expr(v, suffix, ind_vars,
                           'Lstix', ["_spd_"+ss+"[0,0] - 1j*_spd_"+ss+"[0,1]"])
