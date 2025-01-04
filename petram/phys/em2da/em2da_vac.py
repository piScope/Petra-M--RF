'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
import numpy as np

from petram.phys.phys_model  import PhysCoefficient, PhysConstant
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_Vac')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable   
data =  (('epsilonr', VtableElement('epsilonr', type='complex',
                                     guilabel = 'epsilonr',
                                     default = 1.0, 
                                     tip = "relative permittivity" )),
         ('mur', VtableElement('mur', type='complex',
                                     guilabel = 'mur',
                                     default = 1.0, 
                                     tip = "relative permeability" )),
         ('sigma', VtableElement('sigma', type='complex',
                                     guilabel = 'sigma',
                                     default = 0.0, 
                                     tip = "contuctivity" )),
         ('t_mode', VtableElement('t_mode', type='float',
                                     guilabel = 'm',
                                     default = 0.0, 
                                     tip = "mode number" )),)

class Epsilon_o_r(PhysCoefficient):
   '''
    -1j * omega^2 * epsilon0 * epsilonr
   '''
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon_o_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0
       v = super(Epsilon_o_r, self).EvalValue(x)
       v = - v * epsilon0 * self.omega * self.omega /x[0]
       if self.real:  return v.real
       else: return v.imag
       
class Epsilon_x_r(PhysCoefficient):
   '''
    -1j * omega^2 * epsilon0 * epsilonr
   '''
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Epsilon_x_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0
       v = super(Epsilon_x_r, self).EvalValue(x)
       v = - v * epsilon0 * self.omega * self.omega * x[0]
       if self.real:  return v.real
       else: return v.imag
       
class Sigma_o_r(PhysCoefficient):
   '''
    -1j * omega * sigma
   '''
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma_o_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0
       v = super(Sigma_o_r, self).EvalValue(x)
       v = -1j * self.omega * v/x[0]
       if self.real:  return v.real
       else: return v.imag
       
class Sigma_x_r(PhysCoefficient):
   '''
    -1j * omega * sigma
   '''
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Sigma_x_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0
       v = super(Sigma_x_r, self).EvalValue(x)
       v = -1j * self.omega * v * x[0]
       if self.real:  return v.real
       else: return v.imag

class InvMu_x_r(PhysCoefficient):
   '''
      r/mu0/mur
   '''
   def __init__(self, *args, **kwargs):
       super(InvMu_x_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0      
       v = super(InvMu_x_r, self).EvalValue(x)
       v = 1/mu0/v*x[0]
       if self.real:  return v.real
       else: return v.imag
       
class InvMu_o_r(PhysCoefficient):
   '''
      1j/mu0/mur/r
   '''
   def __init__(self, *args, **kwargs):
       self.tmode = kwargs.pop('tmode', 1.0)      
       super(InvMu_o_r, self).__init__(*args, **kwargs)
  
   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0      
       v = super(InvMu_o_r, self).EvalValue(x)
       v = 1/mu0/v/x[0]
       if self.real:  return v.real
       else: return v.imag

class iInvMu_m_o_r(PhysCoefficient):
   '''
      1j/mu0/mur/r
   '''
   def __init__(self, *args, **kwargs):
       self.tmode = kwargs.pop('tmode', 1.0)      
       super(iInvMu_m_o_r, self).__init__(*args, **kwargs)
  
   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0      
       v = super(iInvMu_m_o_r, self).EvalValue(x)
       v = 1j/mu0/v/x[0]*self.tmode
       if self.real:  return v.real
       else: return v.imag
       
class InvMu_m2_o_r(PhysCoefficient):
   '''
      1./mu0/mur/r/m^2
   '''
   def __init__(self, *args, **kwargs):
       self.tmode = kwargs.pop('tmode', 1.0)      
       super(InvMu_m2_o_r, self).__init__(*args, **kwargs)
  
   def EvalValue(self, x):
       from .em2da_const import mu0, epsilon0      
       v = super(InvMu_m2_o_r, self).EvalValue(x)
       v = 1/mu0/v/x[0]*self.tmode*self.tmode
       if self.real:  return v.real
       else: return v.imag
       
def domain_constraints():
   return [EM2Da_Vac]
       
class EM2Da_Vac(EM2Da_Domain):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']
    
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
        return [(0, 1, 1, 1), (1, 0, 1, 1)]

    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        from .em2da_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s, tmode = self.vt.make_value_or_expression(self)
        if not isinstance(e, str): e = str(e)
        if not isinstance(m, str): m = str(m)
        if not isinstance(s, str): s = str(s)

        if kfes == 0: ## ND element (Epoloidal)
            if real:       
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
            imu_x_r = InvMu_x_r(m,  self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real)
            s_x_r = Sigma_x_r(s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega)
            e_x_r = Epsilon_x_r(e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega)
            
            self.add_integrator(engine, 'mur', imu_x_r,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'epsilonr', e_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            self.add_integrator(engine, 'sigma', s_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            
            if tmode != 0:
                imu_o_r_2 = InvMu_m2_o_r(m,  self.get_root_phys().ind_vars,
                                      self._local_ns, self._global_ns,
                                      real = real, tmode = tmode)
                self.add_integrator(engine, 'mur', imu_o_r_2,
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)
                
        elif kfes == 1: ## ND element (Epoloidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))
            imv_o_r_1 = InvMu_o_r(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real)
            e_o_r = Epsilon_o_r(e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega)
            s_o_r = Sigma_o_r(s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega)
            
            self.add_integrator(engine, 'mur', imv_o_r_1,
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr', e_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
            self.add_integrator(engine, 'sigma', s_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass
        
    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
       
        from .em2da_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()
        e, m, s, tmode = self.vt.make_value_or_expression(self)

        if tmode == 0: return
        if not isinstance(e, str): e = str(e)
        if not isinstance(m, str): m = str(m)
        if not isinstance(s, str): s = str(s)
        
        imv_o_r_3 = iInvMu_m_o_r(m,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, tmode = tmode)

        if r == 1 and c == 0:
            # (-a u_vec, div v_scalar)           
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
        else:
            itg =  mfem.MixedVectorGradientIntegrator
        self.add_integrator(engine, 'mur', imv_o_r_3,
                                mbf.AddDomainIntegrator, itg)
        

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        e, m, s, tmode = self.vt.make_value_or_expression(self)
        
        if len(self._sel_index) == 0: return

        self.do_add_scalar_expr(v, suffix, ind_vars, 'sepsilonr', e, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'ssigma', s, add_diag=3)

        var = ['r', 'phi', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')
        
        add_constant(v, 'm_mode', suffix, np.float64(tmode),
                     domains = self._sel_index,
                     gdomain = self._global_ns)
        
    
