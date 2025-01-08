'''
   external current source
'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.coefficient import VCoeff
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.phys_model import VectorPhysCoefficient
from petram.phys.phys_model import PhysCoefficient
from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_extJ')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('jext', VtableElement('jext', type='complex',
                               guilabel='External J',
                               suffix=('x', 'y', 'z'),
                               default=[0, 0, 0],
                               tip="volumetric external current")),)


#from petram.phys.coefficient import PyComplexVectorSliceCoefficient as ComplexVectorSlice


def jwJ_Coeff(exprs, ind_vars, l, g, omega):
    # iomega x Jext
    fac = 1j * omega
    return VCoeff(3, exprs, ind_vars, l, g, return_complex=True, scale=fac)

def domain_constraints():
   return [EM2D_ExtJ]

class EM2D_ExtJ(EM2D_Domain):
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
            self.set_integrator_realimag_mode(real)

            jwJ = jwJ_Coeff(f_name[0], self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns, omega)

            if kfes == 0:
                #jwJ01 = ComplexVectorSlice(jwJ, [0,1])
                jwJ01 = jwJ[[0, 1]]

                self.add_integrator(engine, 'jext', jwJ01,
                                    b.AddDomainIntegrator,
                                    mfem.VectorFEDomainLFIntegrator)
            else:
                jwJ2 = jwJ[2]
                #jwJ2 = ComplexVectorSlice(jwJ, [2])
                self.add_integrator(engine, 'jext', jwJ2,
                                    b.AddDomainIntegrator,
                                    mfem.DomainLFIntegrator)

        else:
            assert False, "should not come here"
