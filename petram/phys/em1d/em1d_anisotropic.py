'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient
from petram.phys.coefficient import SCoeff, MCoeff
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain
from petram.phys.em1d.em1d_vac import EM1D_Vac

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_Anisotropic')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   
data =  (('epsilonr', VtableElement('epsilonr', type='complex',
                                     guilabel = 'epsilonr',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.eye(3, 3),
                                     tip = "relative permittivity" )),
         ('mur', VtableElement('mur', type='complex',
                                     guilabel = 'mur',
                                     default = 1.0, 
                                     tip = "relative permeability" )),
         ('sigma', VtableElement('sigma', type='complex',
                                     guilabel = 'sigma',
                                     suffix =[('x', 'y', 'z'), ('x', 'y', 'z')],
                                     default = np.zeros((3, 3)),
                                     tip = "contuctivity" )),
         ('ky', VtableElement('ky', type='float',
                                     guilabel = 'ky',
                                     default = 0. ,
                                     no_func = True, 
                                     tip = "wave number in the y direction" )),
         ('kz', VtableElement('kz', type='float',
                                     guilabel = 'kz',
                                     default = 0.0,
                                     no_func = True,                               
                                     tip = "wave number in the z direction" )),)

'''
Expansion of matrix is as follows

                 [e_xx  e_xy  e_zz ][Ex]
                 [                 ][  ]
[Wx, Wy, Wz]  =  [e_yx  e_yy  e_yz ][Ey]
                 [                 ][  ]
                 [e_zx  e_zy  e_zz ][Ez]

'''
from petram.phys.phys_const import mu0, epsilon0

'''
class Epsilon(PhysCoefficient):
   #
   # - omega^2 * epsilon0 * epsilonr
   #
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       self.component = kwargs.pop('component', (0, 0))
       super(Epsilon, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(Epsilon, self).EvalValue(x)
       v = np.array(- v * epsilon0 * self.omega * self.omega, copy=False).reshape(3,3)
       v = v[self.component]
       if self.real:  return v.real
       else: return v.imag

def make_epsilon(*args, **kwargs):
    e = args[0]
    component = kwargs.get('component', None)
    omega = kwargs.get('omega', 1.0)
    real = kwargs.get('real', True)
    if any([isinstance(ee, str) for ee in e]):
        return Epsilon(*args, **kwargs)
    else:
        # conj is ignored..(this doesn't no meaning...)       
        if component is None:
            assert False, "index is not given"
            
        v = args[0][component]
        v = - v * epsilon0 * omega * omega        
        if real:  v = v.real
        else: v = v.imag
        return PhysConstant(float(v))
       
class Sigma(PhysCoefficient):
   #
   # -1j * omega * sigma
   #
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       self.component = kwargs.pop('component', (0, 0))       
       super(Sigma, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(Sigma, self).EvalValue(x)
       v = np.array(-1j * self.omega * v, copy=False).reshape(3,3)
       v = v[self.component]       
       if self.real:  return v.real
       else: return v.imag
       
def make_sigma(*args, **kwargs):
    e = args[0]
    component = kwargs.get('component', None)
    omega = kwargs.get('omega', 1.0)
    real = kwargs.get('real', True)
    if any([isinstance(ee, str) for ee in e]):
        return Sigma(*args, **kwargs)
    else:
        # conj is ignored..(this doesn't no meaning...)       
        if component is None:
            assert False, "index is not given"
            
        v = args[0][component]
        v = -1j * omega * v        
        if real:  v = v.real
        else: v = v.imag
        return PhysConstant(float(v))
'''
def Epsilon_Coeff(exprs, ind_vars, l, g, omega):
    # - omega^2 * epsilon0 * epsilonr
    fac = -epsilon0 * omega * omega * cnorm
    return MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

def Sigma_Coeff(exprs, ind_vars, l, g, omega):
    # v = - 1j * self.omega * v
    fac = - 1j * omega*cnorm
    return MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

def Mu_Coeff(exprs, ind_vars, l, g, omega):
    # v = mu * v
    fac = mu0*cnorm
    #return MCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return SCoeff([exprs], ind_vars, l, g, return_complex=True, scale=fac)
     
def domain_constraints():
   return [EM1D_Anisotropic]

class EM1D_Anisotropic(EM1D_Vac):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']

    def get_coeffs2(self, r, c):
        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        e, m, s, ky, kz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        coeff1 = Epsilon_Coeff(e, ind_vars, l, g, omega, cnorm)
        coeff2 = Mu_Coeff(m, ind_vars, l, g, omega, cnorm)
        coeff3 = Sigma_Coeff(s, ind_vars, l, g, omega, cnorm)

        ec = coeff1[r, c]
        sc = coeff3[r, c]

        return ec, sc, ky, kz

    
    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        
        ec, sc, ky, kz = self.get_coeffs2(kfes, kfes)
        self.set_integrator_realimag_mode(real)

        super(EM1D_Anisotropic, self).add_bf_contribution(engine, a, real=real,
                                                          kfes=kfes, ecsc=(ec, sc))


    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))


        ### adding mu. this part is the same as vacuum
        super(EM1D_Anisotropic, self).add_mix_contribution(engine, mbf, r, c, is_trans,
                                                           real = real)
        
        ec, sc, ky, kz = self.get_coeffs2(r, c) 
        self.set_integrator_realimag_mode(real)       

        self.add_integrator(engine, 'epsilonr', ec, mbf.AddDomainIntegrator,
                            mfem.MixedScalarMassIntegrator)
        self.add_integrator(engine, 'sigma', sc, mbf.AddDomainIntegrator,
                                mfem.MixedScalarMassIntegrator)


    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant
        
        e, m, s, ky, kz = self.vt.make_value_or_expression(self)
        
        if len(self._sel_index) == 0: return

        add_constant(v, 'ky', suffix, np.float64(ky),
                     domains = self._sel_index,
                     gdomain = self._global_ns)
        
        add_constant(v, 'kz', suffix, np.float64(kz),
                     domains = self._sel_index,
                     gdomain = self._global_ns)

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', e)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', s)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')



    
