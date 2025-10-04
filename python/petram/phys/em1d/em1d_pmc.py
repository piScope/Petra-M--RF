from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain

from petram.phys.vtable import VtableElement, Vtable   
data =  (('label1', VtableElement(None, 
                                  guilabel = 'Perfect Magnetic Conductor',
                                  default =   "Hx = 0; Hy = 0",
                                  tip = "this is a natural BC" )),)
def bdry_constraints():
   return [EM1D_PMC]

class EM1D_PMC(EM1D_Bdry):
    is_essential = False
    nlterms = []
    vt  = Vtable(data)          
    
