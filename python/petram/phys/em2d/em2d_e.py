'''
   Essential BC
'''
from petram.phys.coefficient import VCoeff
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Bdry
from petram.phys.phys_model import Phys, VectorPhysCoefficient, PhysCoefficient
from petram.phys.em2d.em2d_base import EM2D_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_E')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('E', VtableElement('E', type='complex',
                            guilabel='Electric field',
                            suffix=('x', 'y', 'z'),
                            default=np.array([0., 0., 0.]),
                            tip="essential BC")),)

'''
class Exy(VectorPhysCoefficient):
   def EvalValue(self, x):
       v = super(Exy, self).EvalValue(x)
       v = np.array((v[0], v[1]))
       if self.real:  return v.real
       else: return v.imag
   
class Ez(PhysCoefficient):
   def __init__(self, *args, **kwargs):
       #kwargs['isArray'] = True
       PhysCoefficient.__init__(self, *args, **kwargs)
   def EvalValue(self, x):
       v = super(Ez, self).EvalValue(x)
       v = v[2]
       if self.real:  return v.real
       else: return v.imag
'''
def bdry_constraints():
   return [EM2D_E]

class EM2D_E(EM2D_Bdry):
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

        Exyz = self.vt.make_value_or_expression(self)[0]
        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        coeff1 = VCoeff(3, Exyz, ind_vars, l, g, return_complex=True)

        if kfes == 0:
            '''
            coeff1 = Exy(2, Exyz,
                        self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real)
            '''
            coeff1 = coeff1[[0, 1]]
            coeff1 = coeff1.get_realimag_coefficient(real)
            gf.ProjectBdrCoefficientTangent(coeff1,
                                            mfem.intArray(bdr_attr))
        elif kfes == 1:
            '''
            coeff1 = Ez(Exyz, self.get_root_phys().ind_vars,
                        self._local_ns, self._global_ns,
                        real = real)
            '''
            coeff1 = coeff1[2]
            coeff1 = coeff1.get_realimag_coefficient(real)
            gf.ProjectBdrCoefficient(coeff1,
                                     mfem.intArray(bdr_attr))
