'''
   Surface Impedance

   On external surface    Z(w) n \times H = n \times n \times E    (1)

   (note)
      Robbin BC is 
          n \times 1/mu curl E + \gamma  n \times n \times E = Q   (2) 

      Since 
            curl E = -dB/dt (Faraday's law)
            B = mu H

      with exp(- j \omega t), (2) becomes 
          j \omega  n \times H + \gamma  n \times n \times E = Q

      Note, we don't consider fero-magnetic material where mu changes
      with hysteresis.

      Therefore we use in (1)
          \gamma = j \omega / Z(\omega) 
          Q = 0
      and, the surface integral 
          - \gamma \int F \dot (n \times n \times E) d\Omega 
                 = \gamma \int F \dot E d\Omega 

      is implemented using mfem::VectorFEMassIntegrator.

      We implement two interface.

      1) Using \sigma, \epsilon, \mu, we define the impedance as

          Z(\omega) = sqrt((j \omega \mu))/(\sigam + j\omega \epsilon))

      or directrly specify Z(\omega).

      As a default value, we use 

         \sigma = 1./1.8e-8 = 5.6e7 (Cu at room temperatrue)
         \epsilonr = 1.0
         \mur = 1.0

      CopyRight (c) 2020-  S. Shiraiwa
''' 
import numpy as np

from petram.phys.phys_model  import Phys, PhysCoefficient
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Z')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable

data1 =  (('impedance', VtableElement('impedance', type='complex',
                                       guilabel = 'Z(w)',
                                       default = 1.0,
                                       tip = "impedance " )),)
data2 = (('epsilonr_bnd', VtableElement('epsilonr_bnd', type='complex',
                                    guilabel = 'epsilonr(bnd)',
                                    default = 1.0, 
                                    tip = "boundary relative permittivity" )),                                    
         ('mur_bnd', VtableElement('mur_bnd', type='complex',
                                guilabel = 'mur',
                                default = 1.0, 
                                tip = "boundary relative permeability" )),
         ('sigma_bnd', VtableElement('sigma_bnd', type='complex',
                                 guilabel = 'sigma',
                                 default = 5.6e7,
                                 tip = "boundary condutivey" )),)


vt1  = Vtable(data1)
vt2  = Vtable(data2)

data3 = (('Z_param', VtableElement('Z_param',
                                   type = 'selectable',
                                   guilabel = '',
                                   choices= ('Impedance', 'e/m/s'),
                                   vtables = (vt1, vt2),
                                   default = 'Impedance')),)

from petram.phys.coefficient import SCoeff
from .em3d_const import mu0, epsilon0

class ImpedanceByEMS(mfem.PyCoefficient):
   def __init__(self, ems,  ind_vars, l, g, omega, real):
       self.e = [SCoeff([ems[0]], ind_vars, l, g, real=True),
                 SCoeff([ems[0]], ind_vars, l, g, real=False)]
       self.m = [SCoeff([ems[1]], ind_vars, l, g, real=True),
                 SCoeff([ems[1]], ind_vars, l, g, real=False)]
       self.s = [SCoeff([ems[2]], ind_vars, l, g, real=True),
                 SCoeff([ems[2]], ind_vars, l, g, real=False)]

       self.omega = omega
       self.real = real
       mfem.PyCoefficient.__init__(self)

   def Eval(self, T, ip):

       e = self.e[0].Eval(T, ip) + 1j* self.e[1].Eval(T, ip)
       m = self.m[0].Eval(T, ip) + 1j* self.m[1].Eval(T, ip)
       s = self.s[0].Eval(T, ip) + 1j* self.s[1].Eval(T, ip)

       omega = self.omega
       z = np.sqrt((1j*omega*mu0*m)/(s + 1j*omega*e*epsilon0))
       gamma = -1j*omega/z
       if self.real:
           return gamma.real
       else:
           return gamma.imag

class ImpedanceByZ(mfem.PyCoefficient):
   def __init__(self, z,  ind_vars, l, g, omega, real):
       self.z = [SCoeff([z[0]], ind_vars, l, g, real=True),
                 SCoeff([z[0]], ind_vars, l, g, real=False)]

       self.omega = omega
       self.real = real
       mfem.PyCoefficient.__init__(self)

   def Eval(self, T, ip):
       z = self.z[0].Eval(T, ip) + 1j* self.z[1].Eval(T, ip)
       omega = self.omega
       gamma = -1j*omega/z
       if self.real:
           return gamma.real
       else:
           return gamma.imag          

def bdry_constraints():
   return [EM3D_Impedance]
        
class EM3D_Impedance(EM3D_Bdry):
    is_essential = False
    vt  = Vtable(data3)

    def has_bf_contribution(self, kfes = 0):
        if kfes != 0: return False
        return True
    
    def add_bf_contribution(self, engine, b, real = True, kfes = 0):
        if kfes != 0: return 
        if real:       
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        freq, omega = self.get_root_phys().get_freq_omega()        
        mode, parameters = self.vt.make_value_or_expression(self)[0]

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns
        
        if mode == 'e/m/s':
            er, mr, s  = parameters
            try:
               z = np.sqrt((1j*omega*mu0*mr)/(s + 1j*omega*er*epsilon0))
               gamma = -1j*omega/z
               coeff = SCoeff([gamma], ind_vars, l, g, real=real)
               #assert False, "cause error"
            except:
               #import traceback
               #traceback.print_exc()
               coeff = ImpedanceByEMS(parameters, ind_vars, l, g, omega, real)

        else:
            z  = parameters[0]
            try:
               gamma = -1j*omega/z
               coeff = SCoeff([gamma], ind_vars, l, g, real=real)
               dprint1("Impedance", z)
               #assert False, "cause error"
            except:
               #import traceback
               #traceback.print_exc()
               coeff = ImpedanceByZ(parameters, ind_vars, l, g, omega, real)
               dprint1("Impedance", parameters)                              
     
        self.add_integrator(engine, 'impedance', coeff,
                            b.AddBoundaryIntegrator,
                            mfem.VectorFEMassIntegrator)
        '''
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddBoundaryIntegrator(mfem.VectorFEDomainLFIntegrator(coeff1))
        '''
       
