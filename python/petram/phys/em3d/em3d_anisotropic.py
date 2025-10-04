'''
   anistropic region:
   note that MFEM3.3 does not support matrix form of coefficeint
   for curl-curl. Needs next update.
'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.coefficient import MCoeff
from petram.mfem_config import use_parallel
import numpy as np
from scipy.linalg import inv
from petram.phys.phys_model import MatrixPhysCoefficient, PhysCoefficient, PhysConstant, PhysMatrixConstant
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Anisotropic')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('epsilonr', VtableElement('epsilonr', type='complex',
                                   guilabel='epsilonr',
                                   suffix=[('x', 'y', 'z'), ('x', 'y', 'z')],
                                   default=np.eye(3, 3),
                                   tip="relative permittivity")),
        ('mur', VtableElement('mur', type='complex',
                              guilabel='mur',
                              suffix=[('x', 'y', 'z'), ('x', 'y', 'z')],
                              default=np.eye(3, 3),
                              tip="relative permeability")),
        ('sigma', VtableElement('sigma', type='complex',
                                guilabel='sigma',
                                suffix=[('x', 'y', 'z'), ('x', 'y', 'z')],
                                default=np.zeros((3, 3)),
                                tip="contuctivity")),)


def Epsilon_Coeff(exprs, ind_vars, l, g, omega, cnorm):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega * cnorm
    coeff = MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff


def Sigma_Coeff(exprs, ind_vars, l, g, omega, cnorm):
    # v = - 1j * self.omega * v
    fac = - 1j * omega * cnorm
    coeff = MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff


def Mu_Coeff(exprs, ind_vars, l, g, omega, cnorm):
    fac = mu0/cnorm
    coeff = MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff


def domain_constraints():
    return [EM3D_Anisotropic]


class EM3D_Anisotropic(EM3D_Domain):
    allow_custom_intorder = True
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

    def get_coeffs(self, real=True):
        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        e, m, s = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff(e, ind_vars, l, g, omega, cnorm)
        coeff2 = Mu_Coeff(m, ind_vars, l, g, omega, cnorm)
        coeff3 = Sigma_Coeff(s, ind_vars, l, g, omega, cnorm)

        return coeff1, coeff2, coeff3

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        freq, omega = self.get_root_phys().get_freq_omega()

        coeff1, coeff2, coeff3 = self.get_coeffs()
        self.set_integrator_realimag_mode(real)

        if self.has_pml():
            coeff1 = self.make_PML_coeff(coeff1)
            coeff2 = self.make_PML_coeff(coeff2)
            coeff3 = self.make_PML_coeff(coeff3)
        coeff2 = coeff2.inv()

        if self.allow_custom_intorder and self.add_intorder != 0:
            fes = a.FESpace()
            geom = fes.GetFE(0).GetGeomType()
            order = fes.GetFE(0).GetOrder()
            isPK = (fes.GetFE(0).Space() == mfem.FunctionSpace.Pk)
            orderw = fes.GetElementTransformation(0).OrderW()
            curlcurl_order = order*2 - 2 if isPK else order*2
            mass_order = orderw + 2*order
            curlcurl_order += self.add_intorder
            mass_order += self.add_intorder

            dprint1("Debug: custom int order. Increment = " +
                    str(self.add_intorder))
            dprint1("  FE order: " + str(order))
            dprint1("  OrderW: " + str(orderw))
            dprint1("  CurlCurlOrder: " + str(curlcurl_order))
            dprint1("  FEMassOrder: " + str(mass_order))

            cc_ir = mfem.IntRules.Get(geom, curlcurl_order)
            ms_ir = mfem.IntRules.Get(geom, mass_order)
        else:
            cc_ir = None
            ms_ir = None

        self.add_integrator(engine, 'epsilonr', coeff1,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator,
                            ir=ms_ir)

        if coeff2 is not None:
            self.add_integrator(engine, 'mur', coeff2,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator,
                                ir=cc_ir)
            #coeff2 = self.restrict_coeff(coeff2, engine)
            # a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff2))
        else:
            dprint1("No contrinbution from curlcurl")

        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator,
                            ir=ms_ir)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        if len(self._sel_index) == 0:
            return

        e, m, s = self.vt.make_value_or_expression(self)

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', e)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'mur', m)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', s)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
