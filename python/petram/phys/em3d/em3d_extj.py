'''
   external current source
'''
from petram.phys.coefficient import VCoeff
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.em3d.em3d_base import EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_extJ')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('jext', VtableElement('jext', type='complex',
                               guilabel='External J',
                               suffix=('x', 'y', 'z'),
                               default=[0, 0, 0],
                               tip="volumetric external current")),)

'''
from petram.phys.phys_model  import Phys, VectorPhysCoefficient
class Jext(VectorPhysCoefficient):
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Jext, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       v = super(Jext, self).EvalValue(x)
       v = 1j * self.omega * v
       if self.real:  return v.real
       else: return v.imag
'''


def JextCoeff(exprs, ind_vars, l, g, omega):
    fac = 1j * omega
    coeff = VCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff


def domain_constraints():
    return [EM3D_ExtJ]


class EM3D_ExtJ(EM3D_Domain):
    is_secondary_condition = True
    has_3rd_panel = False
    vt = Vtable(data)

    def has_lf_contribution(self, kfes=0):
        if kfes != 0:
            return False
        return True

    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))

        freq, omega = self.get_root_phys().get_freq_omega()
        f_name = self.vt.make_value_or_expression(self)

        # coeff1 = Jext(3, f_name[0],  self.get_root_phys().ind_vars,
        #                    self._local_ns, self._global_ns,
        #                    real = real, omega = omega)
        self.set_integrator_realimag_mode(real)
        coeff1 = JextCoeff(f_name[0],  self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns, omega)

        self.add_integrator(engine, 'jext', coeff1,
                            b.AddDomainIntegrator,
                            mfem.VectorFEDomainLFIntegrator)

        '''
        coeff1 = self.restrict_coeff(coeff1, engine, vec = True)
        b.AddDomainIntegrator(mfem.VectorFEDomainLFIntegrator(coeff1))
        '''
