'''
  continuity is a condition for internail
  boundary. 
  It has no contribution to weak form
'''
from petram.model import Domain, Bdry, Pair
from petram.phys.phys_cont  import PhysContinuity

def bdry_constraints():
   return [EM2D_Continuity]

class EM2D_Continuity(PhysContinuity):
    pass
