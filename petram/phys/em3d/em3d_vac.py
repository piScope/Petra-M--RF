'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.coefficient import SCoeff
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.phys_model import PhysCoefficient, PhysConstant
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Vac')

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
                                tip="contuctivity")),)

def Epsilon_Coeff(exprs, ind_vars, l, g, omega):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def Sigma_Coeff(exprs, ind_vars, l, g, omega):
    # v = - 1j * self.omega * v
    fac = - 1j * omega
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def Mu_Coeff(exprs, ind_vars, l, g, omega):
    # v = mu * v
    fac = mu0
    coeff = SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff

def domain_constraints():
   return [EM3D_Vac]

class EM3D_Vac(EM3D_Domain):
    vt = Vtable(data)
    #nlterms = ['epsilonr']

    def get_possible_child(self):
        from .em3d_pml import EM3D_LinearPML
        return [EM3D_LinearPML]

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff([e], ind_vars, l, g, omega)
        coeff2 = Mu_Coeff([m], ind_vars, l, g, omega)
        coeff3 = Sigma_Coeff([s], ind_vars, l, g, omega)

        dprint1("epsr, mur, sigma " + str(coeff1) +
                " " + str(coeff2) + " " + str(coeff3))

        return coeff1, coeff2, coeff3

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        # e, m, s
        coeff1, coeff2, coeff3 = self.get_coeffs()
        self.set_integrator_realimag_mode(real)

        if self.has_pml():
            coeff1 = self.make_PML_coeff(coeff1)
            coeff2 = self.make_PML_coeff(coeff2)
            coeff3 = self.make_PML_coeff(coeff3)
            coeff2 = coeff2.inv()
        else:
            coeff2 = coeff2**(-1)

        self.add_integrator(engine, 'epsilonr', coeff1,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)
        self.add_integrator(engine, 'mur', coeff2,
                            a.AddDomainIntegrator,
                            mfem.CurlCurlIntegrator)
        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        if len(self._sel_index) == 0:
            return

        e, m, s = self.vt.make_value_or_expression(self)

        self.do_add_scalar_expr(v, suffix, ind_vars,
                                'sepsilonr', e, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'ssigma', s, add_diag=3)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

