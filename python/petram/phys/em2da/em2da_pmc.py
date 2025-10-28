from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Magnetic Conductor',
                                  default =   "Ht = 0",
                                  tip = "this is a natural BC" )),)
def bdry_constraints():
   return [EM2Da_PMC]

class EM2Da_PMC(EM2Da_Bdry):
    is_essential = False
    nlterms = []
    vt  = Vtable(data)          
    
