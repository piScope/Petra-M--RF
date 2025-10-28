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
data =  (('jext', VtableElement('jext', type='complex',
                             guilabel = 'External J',
                             suffix =('x', 'y', 'z'),
                             default = [0,0,0],
                             tip = "volumetric external current" )),)

from petram.phys.coefficient import VCoeff
#from petram.phys.coefficient import PyComplexVectorSliceCoefficient as ComplexVectorSlice
   
def jwJ_Coeff(exprs, ind_vars, l, g, omega):
    # iomega x Jext
    fac =  1j * omega
    return VCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

def domain_constraints():
   return [EM1D_ExtJ]
 
class EM1D_ExtJ(EM1D_Domain):
    is_secondary_condition = True  # does not count this class for persing "remaining"
    has_3rd_panel = False

    vt  = Vtable(data)
    
    def has_lf_contribution(self, kfes = 0):
        if kfes == 0: return True
        if kfes == 1: return True
        if kfes == 2: return True
        return False

    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes < 3:
            if real:       
                dprint1("Add LF contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add LF contribution(imag)" + str(self._sel_index))
            freq, omega = self.get_root_phys().get_freq_omega()
            self.set_integrator_realimag_mode(real)

            jext = self.vt.make_value_or_expression(self)            
            jwJ = jwJ_Coeff(jext[0], self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns, omega)
            

            if kfes in [0, 1, 2]:
                #  j_slice = ComplexVectorSlice(jwJ, [kfes])
                j_slice = jwJ[kfes]
                self.add_integrator(engine, 'jext', j_slice,
                                    b.AddDomainIntegrator,
                                    mfem.DomainLFIntegrator)

        else:
           assert False, "should not come here"

               
            



        
        
        

