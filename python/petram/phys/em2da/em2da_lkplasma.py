'''
   local-K plasma:
'''
from petram.phys.common.rf_dispersion_lkplasma import (vtable_data0,
                                                       default_kpe_option,
                                                       kpe_options,
                                                       term_options,
                                                       default_term_option)

from petram.phys.common.rf_dispersion_coldplasma import (col_model_options,
                                                         default_col_model,)


from petram.phys.phys_const import mu0, epsilon0
from petram.phys.numba_coefficient import (func_to_numba_coeff_scalar,
                                           func_to_numba_coeff_vector,
                                           func_to_numba_coeff_matrix)
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.phys_model import PhysCoefficient, VectorPhysCoefficient
from petram.phys.phys_model import MatrixPhysCoefficient, Coefficient_Evaluator
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_LocalKPlasma')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

vtable_data = vtable_data0.copy()
vtable_data.extend([
    ('t_mode', VtableElement('t_mode', type="float",
                             guilabel='m',
                             default=0.0,
                             tip="mode number")), ])


'''
Expansion of matrix is as follows

               [e_rz  e_12 ][Erz ]
[Wrz, Wphx] =  [           ][    ] = Wrz e_rz Erz + Wrz e_12 Ephi
               [e_21  e_phi][Ephx]

                                + Wphi e_21 Erz + Wphi*e_phi*Ephi


  Erz = Er e_r + Ez e_z
  Ephx = rho Ephi
'''

kpe_alg_options = ["std", "em2da"]


def domain_constraints():
    return [EM2Da_LocalKPlasma]


class EM2Da_LocalKPlasma(EM2Da_Domain):
    vt = Vtable(vtable_data)

    @classmethod
    def fancy_menu_name(cls):
        return "HotPlasma(local-k)"

    @classmethod
    def fancy_tree_name(cls):
        return "LocalKPlasma"

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        elif kfes == 1:
            return True
        else:
            return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag1, flag2 of MixedBilinearForm
           flag1 : take transpose
           flag2 : take conj
        '''
        return [(0, 1, 1, 1), (1, 0, 1, 1), ]  # (0, 1, -1, 1)]

    def attribute_set(self, v):
        EM2Da_Domain.attribute_set(self, v)
        v["kpe_mode"] = default_kpe_option
        v["kpe_alg"] = kpe_alg_options[0]
        v["col_model"] = default_col_model
        v["lk_terms"] = None
        return v

    def config_terms(self, evt):
        from petram.phys.common.rf_lk_term_panel import ask_rf_lk_terms

        self.vt.preprocess_params(self)
        #B, dens_e, t_e, dens_i, t_i, t_c, masses, charges, kpakpe, kpevec, mmode
        ret = self.vt.make_value_or_expression(self)
        charges = ret[7]

        num_ions = len(charges)
        win = evt.GetEventObject()
        value = ask_rf_lk_terms(win, num_ions, self.lk_terms)
        self.lk_terms = value

    def panel1_param(self):
        panels = super(EM2Da_LocalKPlasma, self).panel1_param()
        panels.append(["kpe mode", None, 1, {"values": kpe_options}])
        panels.append(["kpe alg.", None, 1, {"values": kpe_alg_options}])
        panels.append(["col. model", None, 1, {"values": col_model_options}])
        panels.extend([  # ["contributions", "", 2, None],
            [None, None, 341, {"label": "Customize terms",
                               "func": "config_terms",
                               "sendevent": True,
                               "noexpand": True}], ])

        return panels

    def get_panel1_value(self):
        values = super(EM2Da_LocalKPlasma, self).get_panel1_value()
        values.extend([self.kpe_mode, self.kpe_alg, self.col_model, self])
        return values

    def import_panel1_value(self, v):
        check = super(EM2Da_LocalKPlasma, self).import_panel1_value(v[:-4])
        self.kpe_mode = v[-4]
        self.kpe_alg = v[-3]
        self.col_model = v[-2]
        #self.lk_terms = v[-2]
        return check

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        B, dens_e, t_e, dens_i, t_i, t_c, masses, charges, kpakpe, kpevec, mmode = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars
        kpe_mode = self.kpe_mode
        kpe_alg = self.kpe_alg

        from petram.phys.common.rf_dispersion_lkplasma import value2flags

        num_ions = len(charges)
        terms = value2flags(num_ions, self.lk_terms)

        from petram.phys.common.rf_dispersion_lkplasma import build_coefficients
        coeff1, coeff2, coeff3 = build_coefficients(ind_vars, omega, B, t_c,  dens_e, t_e,
                                                            dens_i, t_i, masses, charges, kpakpe, kpevec,
                                                            kpe_mode, self.col_model, cnorm,
                                                            self._global_ns, self._local_ns,
                                                            kpe_alg=kpe_alg, sdim=2, mmode=mmode,
                                                            terms=terms)

        return coeff1, coeff2, coeff3, mmode

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        coeff1, coeff2, coeff3, mmode = self.jited_coeff
        self.set_integrator_realimag_mode(real)

        invmu = coeff2.inv()
        coeff4 = coeff1 + coeff3

        eps11 = coeff4[[0, 2], [0, 2]]
        eps22 = coeff4[1, 1]

        def invmu_x_r(ptx, invmu):
            return invmu[0, 0]*ptx[0]

        def invmu_o_r(ptx, invmu):
            return invmu[0, 0]/ptx[0]

        def invmu_o_r_tt(ptx, invmu):
            return invmu[0, 0]/ptx[0]*mmode*mmode

        def eps11_x_r(ptx, eps11_):
            return eps11_*ptx[0]

        def eps22_o_r(ptx, eps22_):
            return eps22_/ptx[0]

        if kfes == 0:  # ND element (Epoloidal)
            if real:
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))

            se_x_r = func_to_numba_coeff_matrix(eps11_x_r, shape=(2, 2),
                                                complex=True,
                                                dependency=(eps11,))
            imu_x_r = func_to_numba_coeff_scalar(invmu_x_r,
                                                 complex=True,
                                                 dependency=(invmu,))

            self.add_integrator(engine, 'mur', imu_x_r,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'sigma_epsilonr', se_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)

            if mmode != 0:
                imu_o_r_2 = func_to_numba_coeff_scalar(invmu_o_r_tt,
                                                       complex=True,
                                                       dependency=(invmu,))

                self.add_integrator(engine, 'mur', imu_o_r_2,
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)

        elif kfes == 1:  # H1 element (Etoroidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))

            se_o_r = func_to_numba_coeff_scalar(eps22_o_r,
                                                complex=True,
                                                dependency=(eps22,))
            imv_o_r_1 = func_to_numba_coeff_scalar(invmu_o_r,
                                                   complex=True,
                                                   dependency=(invmu,))

            self.add_integrator(engine, 'mur', imv_o_r_1,
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr', se_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real=True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))

        coeff1, coeff2, coeff3, mmode = self.jited_coeff
        self.set_integrator_realimag_mode(real)

        invmu = coeff2.inv()
        coeff4 = coeff1 + coeff3

        eps12 = coeff4[[0, 2], 1]
        eps21 = coeff4[1, [0, 2]]

        def iinvmu_o_r_t(ptx, invmu):
            return 1j*invmu[0, 0]/ptx[0]*mmode
        imv_o_r_3 = func_to_numba_coeff_scalar(iinvmu_o_r_t,
                                               complex=True,
                                               dependency=(invmu,))

        se_21 = eps21
        se_12 = eps12

        if r == 1 and c == 0:

            # if  is_trans:
            # (a_vec dot u_vec, v_scalar)
            itg = mfem.MixedDotProductIntegrator
            self.add_integrator(engine, 'sigma_epsilon_21', se_21,
                                mbf.AddDomainIntegrator, itg)
            # (-a u_vec, div v_scalar)
            itg = mfem.MixedVectorWeakDivergenceIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                                mbf.AddDomainIntegrator, itg)

        else:
            itg = mfem.MixedVectorProductIntegrator
            self.add_integrator(engine, 'sigma_epsilon_12', se_12,
                                mbf.AddDomainIntegrator, itg)

            # (a grad u_scalar, v_vec)

            itg = mfem.MixedVectorGradientIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                                mbf.AddDomainIntegrator, itg)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        if len(self._sel_index) == 0:
            return

        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, t_i, t_c, masses, charges, kpakpe, kpevec, mmode = self.vt.make_value_or_expression(
            self)

        kpe_mode = self.kpe_mode
        kpe_alg = self.kpe_alg

        add_constant(v, 'm_mode', suffix, np.float64(mmode),
                     domains=self._sel_index,
                     gdomain=self._global_ns)

        from petram.phys.common.rf_dispersion_lkplasma import value2flags

        num_ions = len(charges)
        terms = value2flags(num_ions, self.lk_terms)

        from petram.phys.common.rf_dispersion_lkplasma import build_variables

        ss = self.parent.parent.name()+'_'+self.name()  # phys module name + name
        ret = build_variables(v, ss, ind_vars,
                              omega, B, t_c,  dens_e, t_e,
                              dens_i, t_i, masses, charges,
                              kpakpe, kpevec, kpe_mode, kpe_alg, self.col_model,
                              self._global_ns, self._local_ns,
                              sdim=2, terms=terms)

        from petram.phys.common.rf_dispersion_lkplasma import add_domain_variables_common
        add_domain_variables_common(self, ret, v, suffix, ind_vars)

        var = ['r', 'phi', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        return
