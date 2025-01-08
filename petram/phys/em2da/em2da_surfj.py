'''
   Surface Current Boundary Condition

   On external surface    n \times H = J_surf
   On internal surface    n \times (H1 - H2) = J_surf (not tested)

   (note)
    In MKSA,  1/mu curl E = -dB/dt 1/mu= i omega B / mu .
      n \times 1/mu curl E = n \times -dB/dt 1/mu
                           = n \times i omega H 
                           = i omega J_surf
    
 
    Therefore, surface integral 
        \int W \dot (n \times 1/mu curl E) d\Omega
    becomes
         \int W \dot (i omega J_surf) d\Omega


   CopyRight (c) 2016-  S. Shiraiwa
''' 
import numpy as np

from petram.phys.phys_const  import mu0, epsilon0

from petram.phys.phys_model  import (VectorPhysCoefficient,
                                     PhysCoefficient)
from petram.phys.em2da.em2da_base import EM2Da_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_SurfJ')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable      
data =  (('surfJ', VtableElement('surfJ', type='complex',
                             guilabel = 'Surface J',
                             suffix =('r', 'phi', 'z'),
                             default = [0,0,0],
                             tip = "surface current" )),)
   
class rJsurfp(VectorPhysCoefficient):
   def __init__(self, *args, **kwargs):
       omega = kwargs.pop('omega', 1.0)
       self.mur = kwargs.pop('mur', 1.0)
       self.fac = -1j*omega
       super(rJsurfp, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(rJsurfp, self).EvalValue(x)
       v = np.array((v[0], v[2]))
       v = self.fac * v * x[0]
       if self.real:  return v.real
       else: return v.imag

class Jsurft(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       omega = kwargs.pop('omega', 1.0)
       self.mur = kwargs.pop('mur', 1.0)
       self.fac = -1j*omega
       super(rJsurfp, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(Jsurft, self).EvalValue(x)
       v = self.fac * v[1]
       if self.real:  return v.real
       else: return v.imag
       
def bdry_constraints():
   return [EM2Da_SurfJ]

class EM2Da_SurfJ(EM2Da_Bdry):
    is_essential = False
    vt  = Vtable(data)

    def has_lf_contribution(self, kfes = 0):
        if kfes != 0: return False
        return True
    
    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes != 0: return 
        if real:       
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))

        freq, omega = self.get_root_phys().get_freq_omega()        
        f_name = self.vt.make_value_or_expression(self)


        if kfes == 0:
            coeff1 = rJsurfp(2, f_name[0],  self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real = real, omega = omega,)
            self.add_integrator(engine, 'surfJ', coeff1,
                                b.AddBoundaryIntegrator,
                                mfem.VectorFEDomainLFIntegrator)

        elif kfes == 1:
            coeff1 = Jsurft(f_name[0],  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega,)
            self.add_integrator(engine, 'surfJ', coeff1,
                                b.AddBoundaryIntegrator,
                                mfem.DomainLFIntegrator)



