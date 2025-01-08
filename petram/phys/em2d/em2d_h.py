'''
   Tangential H Boundary Condition

   (note)
    In MKSA,  1/mu curl E = -dH/dt = i omega H.
 
    Therefore, surface integral 
        \int W \dot (n \times 1/mu curl E) d\Omega
    becomes
        \int W \dot (n \times i omega H) d\Omega

    and VectorFEBoundaryTangentLFIntegrator can be used.

   CopyRight (c) 2016-  S. Shiraiwa
''' 

import numpy as np

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys, VectorPhysCoefficient, PhysCoefficient
from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_H')

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
       from petram.phys.phys_const import mu0, epsilon0, c
       self.fac = 1j*omega #/mur
       super(Ht, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from petram.phys.phys_const import mu0, epsilon0, c      

       v = super(Ht, self).EvalValue(x)
       v = self.fac * v[:2]
       if self.real:  return v.real
       else: return v.imag

class Hz(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       omega = kwargs.pop('omega', 1.0)
       from petram.phys.phys_const import mu0, epsilon0, c
       self.fac = 1j*omega #/mur
       super(Hz, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       from petram.phys.phys_const import mu0, epsilon0, c      

       v = super(Hz, self).EvalValue(x)
       v = self.fac * v[2]
       if self.real:  return v.real
       else: return v.imag
       
def bdry_constraints():
   return [EM2D_H]

class EM2D_H(EM2D_Bdry):
    is_essential = False
    vt  = Vtable(data)
    
    def has_lf_contribution(self, kfes=0):
        return True
    
    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        if real:       
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))
            
        from petram.phys.phys_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()        

        h = self.vt.make_value_or_expression(self)
        if kfes == 0:
            coeff1 = Ht(2, h[0],  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, omega = omega)
            self.add_integrator(engine, 'H', coeff1,
                                b.AddBoundaryIntegrator,
                                mfem.VectorFEBoundaryTangentLFIntegrator)
        else:
            coeff1 = Hz(h[0],  self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real, omega = omega)
            self.add_integrator(engine, 'H', coeff1,
                                b.AddBoundaryIntegrator,
                                mfem.BoundaryLFIntegrator)
           
        '''
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddBoundaryIntegrator(mfem.VectorFEBoundaryTangentLFIntegrator(coeff1))
        '''
