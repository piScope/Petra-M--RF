'''
   Essential BC
'''
from petram.phys.numba_coefficient import func_to_numba_coeff_scalar
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.coefficient import VCoeff
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Bdry
from petram.phys.phys_model import Phys, VectorPhysCoefficient, PhysCoefficient
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_E')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


data = (('E', VtableElement('E', type='complex',
                            guilabel='Electric field',
                            suffix=('r', 'phi', 'z'),
                            default=np.array([0, 0, 0]),
                            tip="essential BC")),)


class Ep(VectorPhysCoefficient):
    def EvalValue(self, x):
        v = super(Ep, self).EvalValue(x)
        v = np.array((v[0], v[2]))
        if self.real:
            return v.real
        else:
            return v.imag


class rEt(PhysCoefficient):
    def __init__(self, *args, **kwargs):
        #kwargs['isArray'] = True
        PhysCoefficient.__init__(self, *args, **kwargs)

    def EvalValue(self, x):
        v = super(rEt, self).EvalValue(x)
        v = v[1]*x[0]
        if self.real:
            return v.real
        else:
            return v.imag


def bdry_constraints():
    return [EM2Da_E]


class EM2Da_E(EM2Da_Bdry):
    has_essential = True
    vt = Vtable(data)

    def get_essential_idx(self, kfes):
        if kfes > 2:
            return
        return self._sel_index

    def apply_essential(self, engine, gf, real=False, kfes=0):
        if kfes > 1:
            return
        if real:
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))

        Erphiz = self.vt.make_value_or_expression(self)[0]

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        coeff1 = VCoeff(3, Erphiz, ind_vars, l, g, return_complex=True)

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()

        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        if kfes == 0:
            # coeff1 = Ep(2, Erphiz,
            #            self.get_root_phys().ind_vars,
            #            self._local_ns, self._global_ns,
            #            real = real)
            coeff1 = coeff1[[0, 2]]
            coeff1 = coeff1.get_realimag_coefficient(real)
            gf.ProjectBdrCoefficientTangent(coeff1,
                                            mfem.intArray(bdr_attr))
        elif kfes == 1:
            # coeff1 = rEt(Erphiz, self.get_root_phys().ind_vars,
            #            self._local_ns, self._global_ns,
            #            real = real)
            def r(ptx,):
                return ptx[0]
            r_coeff = func_to_numba_coeff_scalar(r,)

            coeff2 = coeff1[1]
            coeff3 = r_coeff*coeff2
            coeff3 = coeff3.get_realimag_coefficient(real)

            gf.ProjectBdrCoefficient(coeff3,
                                     mfem.intArray(bdr_attr))
