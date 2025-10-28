'''
   external current source
'''
from petram.phys.phys_model import PhysCoefficient
from petram.phys.phys_model import VectorPhysCoefficient
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.coefficient import VCoeff, SCoeff
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain
from petram.phys.numba_coefficient import (func_to_numba_coeff_scalar,
                                           func_to_numba_coeff_vector,
                                           func_to_numba_coeff_matrix)

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_extJ')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('jext', VtableElement('jext', type='complex',
                               guilabel='External J',
                               suffix=('r', 'phi', 'z'),
                               default=[0, 0, 0],
                               tip="volumetric external current")),)


'''
class rJext_p(VectorPhysCoefficient): # i \omega r Jext_t
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(rJext_p, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       v = super(rJext_p, self).EvalValue(x)
       v = np.array((v[0], v[2]))
       v = 1j * self.omega * v * x[0]
       if self.real:  return v.real
       else: return v.imag
       
class Jext_t(PhysCoefficient):  # i \omega Jext_phi
   def __init__(self, *args, **kwargs):
       self.omega = kwargs.pop('omega', 1.0)
       super(Jext_t, self).__init__(*args, **kwargs)
   def EvalValue(self, x):
       v = super(Jext_t, self).EvalValue(x)
       v = 1j * self.omega * v[1]
       if self.real:  return v.real
       else: return v.imag
'''


def JextCoeff(exprs, ind_vars, l, g, omega):
    fac = 1j * omega
    coeff = VCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)
    return coeff


def domain_constraints():
    return [EM2Da_ExtJ]


class EM2Da_ExtJ(EM2Da_Domain):
    is_secondary_condition = True  # does not count this class for persing "remaining"
    has_3rd_panel = False
    vt = Vtable(data)

    def has_lf_contribution(self, kfes=0):
        if kfes > 2:
            return False
        return True

    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        if kfes < 2:
            if real:
                dprint1("Add LF contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add LF contribution(imag)" + str(self._sel_index))
            freq, omega = self.get_root_phys().get_freq_omega()
            f_name = self.vt.make_value_or_expression(self)
            ind_vars = self.get_root_phys().ind_vars

            jext = JextCoeff(f_name[0], ind_vars, self._local_ns, self._global_ns,
                             omega)

            def rjext_p(ptx, jext):
                return np.array((jext[0]*ptx[0], jext[2]*ptx[0]), dtype=np.complex128)

            def jext_t(ptx, jext):
                return jext[1]

            if kfes == 0:
                '''
                coeff1 = rJext_p(2, f_name[0],  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
                '''
                coeff1 = func_to_numba_coeff_vector(rjext_p,
                                                    shape=(2, ),
                                                    complex=True,
                                                    dependency=(jext,))

                self.add_integrator(engine, 'jext', coeff1,
                                    b.AddDomainIntegrator,
                                    mfem.VectorFEDomainLFIntegrator)
            else:
                '''
                coeff1 = Jext_t(f_name[0],  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real, omega = omega)
                '''
                coeff1 = func_to_numba_coeff_scalar(jext_t, complex=True,
                                                    dependency=(jext,))
                self.add_integrator(engine, 'jext', coeff1,
                                    b.AddDomainIntegrator,
                                    mfem.DomainLFIntegrator)

        else:
            assert False, "should not come here"
