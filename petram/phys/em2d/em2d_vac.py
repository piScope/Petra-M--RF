'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain, EM2D_Domain_helper
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.coefficient import SCoeff
from petram.mfem_config import use_parallel
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_Vac')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

'''   
from petram.phys.coefficient import PyComplexPowCoefficient as ComplexPow
from petram.phys.coefficient import PyComplexProductCoefficient as ComplexProduct
from petram.phys.coefficient import PyComplexSumCoefficient as ComplexSum

from petram.phys.coefficient import PyComplexMatrixSumCoefficient as ComplexMatrixSum
from petram.phys.coefficient import PyComplexMatrixInvCoefficient as ComplexMatrixInv
from petram.phys.coefficient import PyComplexMatrixSliceCoefficient as ComplexMatrixSlice
from petram.phys.coefficient import PyComplexMatrixProductCoefficient as ComplexMatrixProduct
from petram.phys.coefficient import PyComplexMatrixAdjCoefficient as ComplexMatrixAdj
'''
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
        ('kz', VtableElement('Nz', type='float',
                             guilabel='kz',
                             default=0.0,
                             no_func=True,
                             tip="out-of-plane wave number")),)


def Epsilon_Coeff(exprs, ind_vars, l, g, omega):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega
    return SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)


def Sigma_Coeff(exprs, ind_vars, l, g, omega):
    # v = - 1j * self.omega * v
    fac = - 1j * omega
    return SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)


def Mu_Coeff(exprs, ind_vars, l, g, omega):
    # v = mu * v
    fac = mu0
    return SCoeff(exprs, ind_vars, l, g, return_complex=True, scale=fac)

def domain_constraints():
   return [EM2D_Vac]

class EM2D_Vac(EM2D_Domain, EM2D_Domain_helper):
    vt = Vtable(data)
    #nlterms = ['epsilonr']

    def get_possible_child(self):
        from .em2d_pml import EM2D_LinearPML
        return [EM2D_LinearPML]

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
        return [(0, 1, 1, 1), (1, 0, 1, 1)]

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s, kz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff([e], ind_vars, l, g, omega)
        coeff2 = Mu_Coeff([m], ind_vars, l, g, omega)
        coeff3 = Sigma_Coeff([s], ind_vars, l, g, omega)

        #dprint1("epsr, mur, sigma " + str(coeff1) + " " + str(coeff2) + " " + str(coeff3))
        return coeff1, coeff2, coeff3, kz

    def get_coeffs_2(self):
        # e, m, s
        coeff1, coeff2, coeff3, kz = self.get_coeffs()

        if self.has_pml():
            coeff2 = self.make_PML_coeff(coeff2)

            '''
            tmp = ComplexMatrixInv(coeff2)
            mu11 = ComplexMatrixSlice(tmp, [0,1], [0,1])
            mu11 = ComplexMatrixAdj(mu11)
            mu22 = ComplexMatrixSlice(tmp, [2], [2])                        
            k2_over_mu11 =   ComplexMatrixProduct(mu11, kz*kz)
            ik_over_mu11 =   ComplexMatrixProduct(mu11, 1j*kz)
            '''
            tmp = coeff2.inv()
            mu11 = tmp[[0, 1], [0, 1]]
            mu11 = mu11.adj()
            mu22 = tmp[2, 2]
            k2_over_mu11 = mu11*(kz*kz)
            ik_over_mu11 = mu11*(1j*kz)
            mu = [mu11, mu22, k2_over_mu11, ik_over_mu11]

            '''
            coeff4 = ComplexSum(coeff1, coeff3)
            '''
            coeff4 = coeff1 + coeff3
            coeff4 = self.make_PML_coeff(coeff4)

            '''
            eps11 = ComplexMatrixSlice(coeff4, [0,1], [0,1])
            eps21 = ComplexMatrixSlice(coeff4, [0,1], [2])
            eps12 = ComplexMatrixSlice(coeff4, [2], [0,1])
            eps22 = ComplexMatrixSlice(coeff4, [2], [2])
            '''
            eps11 = coeff4[[0, 1], [0, 1]]
            eps21 = coeff4[[0, 1], 2]
            eps12 = coeff4[2, [0, 1]]
            eps22 = coeff4[2, 2]

            eps = [eps11, eps12, eps21, eps22]

        else:
            '''
            coeff2 = ComplexPow(coeff2, -1)
            '''
            coeff2 = coeff2**(-1)
            one_over_mu = coeff2
            '''
            k2_over_mu =   ComplexProduct(coeff2, kz*kz)
            ik_over_mu =   ComplexProduct(coeff2, 1j*kz)
            '''
            k2_over_mu = coeff2*(kz*kz)
            ik_over_mu = coeff2*(1j*kz)

            mu = [one_over_mu, one_over_mu, k2_over_mu, ik_over_mu]

            '''
            coeff4 = ComplexSum(coeff1, coeff3)
            '''
            coeff4 = coeff1 + coeff3

            eps = [coeff4, None, None, coeff4]

        return eps, mu, kz

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        #neg_w2eps, one_over_mu, k2_over_mu, ik_over_mu, neg_iwsigma, kz= self.get_coeffs_2(real)

        eps, mu, kz = self.get_coeffs_2()

        self.set_integrator_realimag_mode(real)
        self.call_bf_add_integrator(eps,  mu, kz, engine, a, kfes)

        '''
        if kfes == 0: ## ND element (Epoloidal)
            if real:       
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
                
            self.add_integrator(engine, 'mur', mu[1],
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'epsilon_sigma', eps[0],
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            

            if kz != 0:
                self.add_integrator(engine, 'mur', mu[2],
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
                
        elif kfes == 1: ## ND element (Epoloidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))

            self.add_integrator(engine, 'mur_2', mu[0],
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr_sigma', esp[3]
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass
        '''

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real=True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) + ')'
                    + str(self._sel_index))
        eps, mu, kz = self.get_coeffs_2()

        self.set_integrator_realimag_mode(real)
        self.call_mix_add_integrator(eps, mu, engine, mbf, r, c, is_trans)

        '''       
        neg_w2eps, one_over_mu, k2_over_mu, ik_over_mu, neg_iwsigma, kz = self.get_coeffs_2(real)
        
        self.set_integrator_realimag_mode(real)
        
        if r == 1 and c == 0:
            # (-a u_vec, div v_scalar)           
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
        else:
            itg =  mfem.MixedVectorGradientIntegrator

        self.add_integrator(engine, 'mur', ik_over_mu,
                                mbf.AddDomainIntegrator, itg)
        '''

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant
        e, m, s, kz = self.vt.make_value_or_expression(self)

        if len(self._sel_index) == 0:
            return

        self.do_add_scalar_expr(v, suffix, ind_vars,
                                'sepsilonr', e, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'ssigma', s, add_diag=3)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains=self._sel_index,
                     gdomain=self._global_ns)

