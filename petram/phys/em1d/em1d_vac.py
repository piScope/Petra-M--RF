'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0

from petram.mfem_config import use_parallel
import numpy as np


from petram.phys.phys_model  import PhysCoefficient, PhysConstant
from petram.phys.coefficient import SCoeff, MCoeff
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_Vac')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('epsilonr', VtableElement('epsilonr', type='complex',
                                   guilabel='epsilonr',
                                   default=1.0,
                                   tip="relative permittivity")),
        ('mur', VtableElement('mur', type='complex',
                              guilabel='mur',
                              default=1.0,
                              tip="relative permeability")),
        ('sigma', VtableElement('sigma', type='complex',
                                guilabel='sigma',
                                default=0.0,
                                tip="contuctivity")),
        ('ky', VtableElement('ky', type='int',
                             guilabel='float',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the y direction")),
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.0,
                             no_func=True,
                             tip="wave number in the z direction")),)


'''       
class Epsilon(PhysCoefficient):
   #
   #  - omega^2 * epsilon0 * epsilonr
   #
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon, self).__init__(*args, **kwargs)

class Sigma(PhysCoefficient):
   #
   # -1j * omega * sigma
   #
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(Sigma, self).EvalValue(x)
       v = -1j * self.omega * v
       if self.real:  return v.real
       else: return v.imag

class InvMu(PhysCoefficient):
   #
   #   1/mu0/mur*(factor)
   #
   def __init__(self, *args, **kwargs):
       self._extra_fac = kwargs.pop("factor", 1.)
       super(InvMu, self).__init__(*args, **kwargs)

    def __init__(self, *args, **kwargs):
        self._extra_fac = kwargs.pop("factor", 1.)
        super(InvMu, self).__init__(*args, **kwargs)

    def EvalValue(self, x):
        v = super(InvMu, self).EvalValue(x)
        v = 1/mu0/v*self._extra_fac

        if self.real:
            return v.real
        else:
            return v.imag

       if self.real:  return v.real
       else: return v.imag
'''
def Epsilon(exprs, ind_vars, l, g, omega=1):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega
    coeff = SCoeff([exprs], ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def Sigma(exprs, ind_vars, l, g, omega=1):
    # - 1j * self.omega * sigma
    fac = - 1j * omega
    coeff = SCoeff([exprs], ind_vars, l, g, return_complex=True, scale=fac)
    return coeff
       
def InvMu(m, ind_vars, l, g, factor=1):
    coeff = SCoeff([m], ind_vars, l, g, return_complex=True, scale=mu0)
    return 1./coeff*factor
       
def domain_constraints():
    return [EM1D_Vac]

class EM1D_Vac(EM1D_Domain):
    vt = Vtable(data)
    #nlterms = ['epsilonr']

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        elif kfes == 1:
            return True
        elif kfes == 2:
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag1, flag2 of MixedBilinearForm
           flag1 : take transpose if negative
           flag2 : take conj if -1
        '''
        return [(0, 1, 1, 1),
                (1, 0, 1, 1),
                (1, 2, 1, 1),
                (2, 1, 1, 1),
                (0, 2, 1, 1),
                (2, 0, 1, 1), ]

    def add_bf_contribution(self, engine, a, real=True, kfes=0,
                            ecsc=None):
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s, ky, kz = self.vt.make_value_or_expression(self)
        if not isinstance(e, str):
            e = str(e)
        if not isinstance(m, str):
            m = str(m)
        if not isinstance(s, str):
            s = str(s)

        txt = ["Ex in ", "Ey in ", "Ez in "][kfes]
        if real:
            dprint1("Add BF contribution(real) " + txt + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + txt + str(self._sel_index))

        if ecsc is None:
            sc = Sigma(s, self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       omega = omega)
            ec = Epsilon(e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                              omega = omega)
        else:
            # anistropic case...
            ec, sc = ecsc

        self.set_integrator_realimag_mode(real)
        
        self.add_integrator(engine, 'epsilonr', sc, a.AddDomainIntegrator,
                            mfem.MassIntegrator)
        self.add_integrator(engine, 'sigma', ec, a.AddDomainIntegrator,
                            mfem.MassIntegrator)
        if kfes == 0:  # Ex
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            factor = ky**2 + kz**2)
            
            self.add_integrator(engine, 'mur', imu, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

        elif kfes == 1 or kfes == 2: # Ey and Ez
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns)

            self.add_integrator(engine, 'mur', imu, a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)

            if kfes == 1: fac = kz*kz
            if kfes == 2: fac = ky*ky            

            imu = InvMu(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            factor = fac)

            self.add_integrator(engine, 'mur', imu, a.AddDomainIntegrator,
                                mfem.MassIntegrator)

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real=True):

        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))

        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s, ky, kz = self.vt.make_value_or_expression(self)

        if not isinstance(e, str):
            e = str(e)
        if not isinstance(m, str):
            m = str(m)
        if not isinstance(s, str):
            s = str(s)

        if r == 0 and c == 1:
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        factor=1j*ky)
            itg = mfem.MixedScalarDerivativeIntegrator
        elif r == 0 and c == 2:
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        factor=1j*kz)
            itg = mfem.MixedScalarDerivativeIntegrator
        elif r == 1 and c == 0:
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        factor=1j*ky)
            itg = mfem.MixedScalarWeakDerivativeIntegrator
        elif r == 1 and c == 2:
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        factor=-ky*kz)
            itg = mfem.MixedScalarMassIntegrator
        elif r == 2 and c == 0:
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        factor=1j*kz)
            itg = mfem.MixedScalarWeakDerivativeIntegrator
        elif r == 2 and c == 1:
            imu = InvMu(m,  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        factor=-ky*kz)
            itg = mfem.MixedScalarMassIntegrator
        else:
            assert False, "Something is wrong..if it comes here;D"

        self.set_integrator_realimag_mode(real)            
        self.add_integrator(engine, 'mur', imu,
                            mbf.AddDomainIntegrator, itg)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        e, m, s, ky, kz = self.vt.make_value_or_expression(self)

        if len(self._sel_index) == 0:
            return

        add_constant(v, 'ky', suffix, np.float64(ky),
                     domains=self._sel_index,
                     gdomain=self._global_ns)

        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains=self._sel_index,
                     gdomain=self._global_ns)

        self.do_add_scalar_expr(v, suffix, ind_vars,
                                'sepsilonr', e, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'ssigma', s, add_diag=3)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
