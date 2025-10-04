'''
   Tangential H Boundary Condition

   (note)
    In MKSA,  1/mu curl E = -dH/dt = i omega H.
 
    Therefore, surface integral 
        \int W \dot (n \times 1/mu curl E) d\Omega
    becomes
        \int W \dot (n \times i omega H) d\Omega

    VectorFEBoundaryTangentLFIntegrator can be used.

   CopyRight (c) 2016-  S. Shiraiwa
''' 

import numpy as np

from petram.phys.phys_const import mu0, epsilon0

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys, VectorPhysCoefficient
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_H')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem
   
from petram.phys.vtable import VtableElement, Vtable      
data =  (('H', VtableElement('H', type='complex',
                             guilabel = 'H',
                             suffix =('x', 'y', 'z'),
                             default = [0,0,0],
                             tip = "magnetic field" )),)

class Ht(VectorPhysCoefficient):
   def __init__(self, *args, **kwargs):
       omega = kwargs.pop('omega', 1.0)
       self.fac = -1j*omega  # /mur
       super(Ht, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(Ht, self).EvalValue(x)
       nv = v*self.fac

       if self.real: return nv.real
       else: return nv.imag


def bdry_constraints():
   return [EM3D_H]
       
class EM3D_H(EM3D_Bdry):
    is_essential = False
    vt  = Vtable(data)
    
    def has_lf_contribution(self, kfes = 0):
        if kfes != 0: return False
        return True
    
    def add_lf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes != 0: return
        if real:       
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))
            
        freq, omega = self.get_root_phys().get_freq_omega()        

        h = self.vt.make_value_or_expression(self)
        coeff1 = Ht(3, h[0],  self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns,
                           real = real, omega = omega)
        self.add_integrator(engine, 'H', coeff1,
                            b.AddBoundaryIntegrator,
                            mfem.VectorFEBoundaryTangentLFIntegrator)
