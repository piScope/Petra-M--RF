from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain
import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_PEC')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Electric Conductor',
                                  default =   "Et = 0",
                                  tip = "Essential Homogenous BC" )),)

def bdry_constraints():
   return [EM2Da_PEC]

class EM2Da_PEC(EM2Da_Bdry):
    has_essential = True
    nlterms = []
    vt  = Vtable(data)          
    
    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        elif kfes == 1:
            return self._sel_index            
        else:
            return []

    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes > 1: return
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

        if kfes == 0:
            coeff1 = mfem.VectorArrayCoefficient(2)
            coeff1.Set(0, mfem.ConstantCoefficient(0.0))
            coeff1.Set(1, mfem.ConstantCoefficient(0.0))            
            gf.ProjectBdrCoefficientTangent(coeff1,
                                            mfem.intArray(bdr_attr))
        elif kfes == 1:
            coeff1 = mfem.ConstantCoefficient(0.0)
            gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))

        


