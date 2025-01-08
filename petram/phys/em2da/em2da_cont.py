'''
  continuity is a condition for internail
  boundary. 
  It has no contribution to weak form
'''
from petram.model import Domain, Bdry, Pair
from petram.phys.phys_cont  import PhysContinuity

def bdry_constraints():
   return [EM2Da_Continuity]

class EM2Da_Continuity(PhysContinuity):
    pass
