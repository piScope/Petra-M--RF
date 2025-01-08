'''`
   Essential BC
'''
import numpy as np

from petram.model import Bdry
from petram.phys.phys_model  import Phys, PhysCoefficient
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_E')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable      
data =  (('E', VtableElement('E', type='complex',
                             guilabel = 'Ey and Ez',
                             suffix =('y','z'),
                             default = np.array([0, 0, ]),
                             tip = "essential BC" )),)

class Et(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       #kwargs['isArray'] = True
       self.Eyz_idx = kwargs.pop('Eyz_idx')
       PhysCoefficient.__init__(self, *args, **kwargs)
   def EvalValue(self, x):
       v = super(Et, self).EvalValue(x)
       v = v[self.Eyz_idx]
       if self.real:  return v.real
       else: return v.imag
       
def bdry_constraints():
   return [EM1D_E]
   
class EM1D_E(EM1D_Bdry):
    has_essential = True
    vt  = Vtable(data)
    
    def get_essential_idx(self, kfes):
        if kfes > 3: return
        return self._sel_index

    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes == 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))
            
        Eyz = self.vt.make_value_or_expression(self)              
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        if kfes == 1:
            coeff1 = Et(Eyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, Eyz_idx=0)
            gf.ProjectBdrCoefficient(coeff1,
                                            mfem.intArray(bdr_attr))
        elif kfes == 2:
            coeff1 = Et(Eyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, Eyz_idx=1)
            gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))






        
        
        

