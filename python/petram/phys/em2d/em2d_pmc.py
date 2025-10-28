from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Magnetic Conductor',
                                  default =   "Ht = 0",
                                  tip = "this is a natural BC" )),)
def bdry_constraints():
   return [EM2D_PMC]

class EM2D_PMC(EM2D_Bdry):
    is_essential = False
    nlterms = []
    vt  = Vtable(data)          
    
