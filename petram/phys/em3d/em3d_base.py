import traceback
import numpy as np

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule, VectorPhysCoefficient

# define variable for this BC.
from petram.phys.vtable import VtableElement, Vtable
data =  (('Einit', VtableElement('Einit', type='float',
                                  guilabel = 'E(init)',
                                  suffix =('x', 'y', 'z'),
                                  default = np.array([0,0,0]), 
                                  tip = "initial_E",
                                  chkbox = True)),)
class Einit(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(Einit, self).EvalValue(x)
       if self.real:  val = v.real
       else: val =  v.imag
       return val

class EM3D_Domain(Domain, Phys):
    has_3rd_panel = True    
    vt3  = Vtable(data)   
    def __init__(self, **kwargs):
        super(EM3D_Domain, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM3D_Domain, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v
    
    def get_init_coeff(self, engine, real=True, kfes=0):
        if kfes != 0: return
        if not self.use_Einit: return
        
        f_name = self.vt3.make_value_or_expression(self)
        coeff = Einit(3, f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
        return self.restrict_coeff(coeff, engine, vec = True)
     
    def has_pml(self):
        from .em3d_pml import EM3D_PML
        for obj in self.walk():
            if isinstance(obj, EM3D_PML) and obj.enabled:
                return True
    def get_pml(self):
        from .em3d_pml import EM3D_PML
        return [obj for obj in self.walk() if isinstance(obj, EM3D_PML) and obj.enabled]
    
    def make_PML_coeff(self, coeff):
        pmls = self.get_pml()
        if len(pmls) > 2: assert False, "Multiple PML is set"
        coeff1 = pmls[0].make_PML_coeff(coeff)
        return coeff1

class EM3D_Bdry(Bdry, Phys):
    has_3rd_panel = True        
    vt3  = Vtable(data)   
    def __init__(self, **kwargs):
        super(EM3D_Bdry, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM3D_Bdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_init_coeff(self, engine, real=True, kfes=0):
        if kfes != 0: return
        if not self.use_Einit: return
        
        f_name = self.vt3.make_value_or_expression(self)
        coeff = Einit(3, f_name[0],
                       self.get_root_phys().ind_vars,
                       self._local_ns, self._global_ns,
                       real = real)
        return self.restrict_coeff(coeff, engine, vec = True)
     

