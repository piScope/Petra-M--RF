'''
   Anistropic media:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma

    Expansion of matrix is as follows

               [e_xy  e_12 ][Exy ]
[Wxy, Wz]   =  [           ][    ] = Wrz e_xy Erz + Wxy e_12 Ez
               [e_21  e_zz][Ez  ]

                                   + Wz e_21 Exy + Wz*e_zz*Exy


  Exy = Ex e_x + Ey e_y
  Ez =  Ez e_z

'''
import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_Anisotropic')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

'''
from petram.phys.coefficient import PyComplexMatrixInvCoefficient as ComplexMatrixInv
from petram.phys.coefficient import PyComplexMatrixProductCoefficient as ComplexMatrixProduct
from petram.phys.coefficient import PyComplexMatrixSumCoefficient as ComplexMatrixSum
from petram.phys.coefficient import PyComplexMatrixSliceCoefficient as ComplexMatrixSlice
from petram.phys.coefficient import PyComplexMatrixAdjCoefficient as ComplexMatrixAdj
'''
from petram.phys.vtable import VtableElement, Vtable   
data =  (('epsilonr', VtableElement('epsilonr', type='complex',
                                     guilabel = 'epsilonr',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.eye(3),
                                     tip = "relative permittivity" )),
         ('mur', VtableElement('mur', type='complex',
                                     guilabel = 'mur',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.eye(3),
                                     tip = "relative permeability" )),
         ('sigma', VtableElement('sigma', type='complex',
                                     guilabel = 'sigma',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.zeros((3, 3)),
                                     tip = "contuctivity" )),
         ('kz', VtableElement('kz', type='float',
                                     guilabel = 'kz',
                                     default = 0.0,
                                     no_func = True,
                                     tip = "out-of-plane wave number" )),)

from petram.phys.coefficient import MCoeff
from petram.phys.phys_const import mu0, epsilon0

def Epsilon_Coeff(exprs, ind_vars, l, g, omega, cnorm):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega * cnorm
    return MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

def Sigma_Coeff(exprs, ind_vars, l, g, omega,  cnorm):
    # v = - 1j * self.omega * v
    fac = - 1j * omega * cnorm
    return MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

def Mu_Coeff(exprs, ind_vars, l, g, omega, cnorm):
    # v = mu * v
    fac = mu0/cnorm
    return MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain, EM2D_Domain_helper

def domain_constraints():
   return [EM2D_Anisotropic]
class EM2D_Anisotropic(EM2D_Domain, EM2D_Domain_helper):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']
    def get_possible_child(self):
        from .em2d_pml      import EM2D_LinearPML
        return [EM2D_LinearPML]
    
    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        elif kfes == 1: return True        
        else: return False
        
    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag1, flag2 of MixedBilinearForm
           flag1 : take transpose
           flag2 : take conj
        '''
        return [(0, 1, 1, 1), (1, 0, 1, 1),]#(0, 1, -1, 1)]

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        e, m, s, kz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        coeff1 = Epsilon_Coeff(e, ind_vars, l, g, omega, cnorm)
        coeff2 = Mu_Coeff(m, ind_vars, l, g, omega, cnorm)
        coeff3 = Sigma_Coeff(s, ind_vars, l, g, omega, cnorm)

        #dprint1("epsr, mur, sigma " + str(coeff1) + " " + str(coeff2) + " " + str(coeff3))
        return coeff1, coeff2, coeff3, kz

    def get_coeffs_2(self):
        # e, m, s
        coeff1, coeff2, coeff3, kz = self.get_coeffs()
        '''
        coeff4 = ComplexMatrixSum(coeff1, coeff3)      # -> coeff4 = coeff1 + coeff3
        '''
        coeff4 = coeff1 + coeff3

        if self.has_pml():
            coeff4 = self.make_PML_coeff(coeff4)
            coeff2 = self.make_PML_coeff(coeff2)

        '''
        eps11 = ComplexMatrixSlice(coeff4, [0,1], [0,1])
        eps21 = ComplexMatrixSlice(coeff4, [0,1], [2])
        eps12 = ComplexMatrixSlice(coeff4, [2], [0,1])
        eps22 = ComplexMatrixSlice(coeff4, [2], [2])
        '''
        eps11 = coeff4[[0,1], [0,1]]
        eps21 = coeff4[[0,1], 2]
        eps12 = coeff4[2, [0,1]]
        eps22 = coeff4[2,2]
        
        eps = [eps11, eps12, eps21, eps22]

        '''
        tmp = ComplexMatrixInv(coeff2)
        '''
        tmp = coeff2.inv()

        '''
        mu11 = ComplexMatrixSlice(tmp, [0,1], [0,1])   #-> mul1 = tmp[[0,1], [0,1]]
        mu11 = ComplexMatrixAdj(mu11)                  #-> mul1 = mul1.adj()
        mu22 = ComplexMatrixSlice(tmp, [2], [2])                        
        k2_over_mu11 =   ComplexMatrixProduct(mu11, kz*kz)  #-> k2_over_mul1 = mul1*(kz*kz)
        ik_over_mu11 =   ComplexMatrixProduct(mu11, 1j*kz)
        '''
        mu11 = tmp[[0,1], [0,1]]
        mu11 = mu11.adj()
        mu22 = tmp[2,2]
        k2_over_mu11 = mu11*(kz*kz)
        ik_over_mu11 = mu11*(1j*kz)
        
        mu = [mu11, mu22, k2_over_mu11, ik_over_mu11]

        return eps, mu, kz
            
    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        #freq, omega = self.get_root_phys().get_freq_omega()
        eps, mu, kz =  self.get_coeffs_2()
        
        self.set_integrator_realimag_mode(real)

        if real:
            if kfes == 0:           
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            elif kfes == 1:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))               
        else:
            if kfes == 0:           
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
            elif kfes == 1:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))               
        
        self.call_bf_add_integrator(eps,  mu, kz,  engine, a, kfes)
        
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
                
        elif kfes == 1: ## H1 element (Etoroidal)
            self.add_integrator(engine, 'mur', mu[0],
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilon_sigma', eps[3],
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass
        '''
    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
       
        #freq, omega = self.get_root_phys().get_freq_omega()
        eps, mu, kz =  self.get_coeffs_2()

        self.set_integrator_realimag_mode(real)        
        self.call_mix_add_integrator(eps, mu, engine, mbf, r, c, is_trans)

        '''
        if r == 1 and c == 0:        
            #if  is_trans:
            # (a_vec dot u_vec, v_scalar)                        
            itg = mfem.MixedDotProductIntegrator
            self.add_integrator(engine, 'epsilon', eps[1],
                                mbf.AddDomainIntegrator, itg)

            # (-a u_vec, div v_scalar)            
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
            self.add_integrator(engine, 'mur', mu[3],
                                mbf.AddDomainIntegrator, itg)
            #print r, c, mbf
        else:
            # (a_vec dot u_scalar, v_vec)
            itg = mfem.MixedVectorProductIntegrator
            self.add_integrator(engine, 'epsilon', eps[2],
                             mbf.AddDomainIntegrator, itg)
            
            # (a grad u_scalar, v_vec)
            itg =  mfem.MixedVectorGradientIntegrator
            self.add_integrator(engine, 'mur', mu[3],
                             mbf.AddDomainIntegrator, itg)
        '''
    def add_domain_variables(self, v, n, suffix, ind_vars):

        from petram.helper.variables import add_constant
        
        e, m, s, kz = self.vt.make_value_or_expression(self)

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', e)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'mur', m)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', s)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
        
        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains = self._sel_index,
                     gdomain = self._global_ns)


    
