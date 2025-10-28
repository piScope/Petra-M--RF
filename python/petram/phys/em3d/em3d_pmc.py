from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Magnetic Conductor',
                                  default =   "Ht = 0",
                                  tip = "this is a natural BC" )),)
def bdry_constraints():
   return [EM3D_PMC]

class EM3D_PMC(EM3D_Bdry):
    is_essential = False
    nlterms = []
    vt  = Vtable(data)          
    
