import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_PEC')

from petram.phys.phys_model  import PhysConstant
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Electric Conductor',
                                  default =   "Ey=0 and Ez=0",
                                  tip = "Essential Homogenous BC" )),)

def bdry_constraints():
   return [EM1D_PEC]

class EM1D_PEC(EM1D_Bdry):
    has_essential = True
    nlterms = []
    vt  = Vtable(data)          
    
    def get_essential_idx(self, kfes):
        if kfes == 1:
            return self._sel_index
        elif kfes == 2:
            return self._sel_index            
        else:
            return []

    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes == 0: return
        
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))
            
        Erphiz = self.vt.make_value_or_expression(self)              
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        coeff =  PhysConstant(0.0)
        gf.ProjectBdrCoefficient(coeff,
                                 mfem.intArray(bdr_attr))


        


