from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_model import Phys
from petram.model import Domain, Bdry, Pair
from petram.mfem_config import use_parallel
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_PEC')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('label1', VtableElement(None,
                                 guilabel='Perfect Electric Conductor',
                                 default="Et = 0",
                                 tip="Essential Homogenous BC")),)

def bdry_constraints():
   return [EM3D_PEC]

class EM3D_PEC(EM3D_Bdry):
    has_essential = True
    nlterms = []
    vt = Vtable(data)

    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, kfes, real=False,
                        **kwargs):

        if kfes != 0:
            return
        if real:
            dprint1("Apply Ess. E=0 (real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess. E=0 (imag)" + str(self._sel_index))

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        if kfes == 0:
            coeff1 = mfem.VectorArrayCoefficient(3)
            coeff1.Set(0, mfem.ConstantCoefficient(0.0))
            coeff1.Set(1, mfem.ConstantCoefficient(0.0))
            coeff1.Set(2, mfem.ConstantCoefficient(0.0))
            gf.ProjectBdrCoefficientTangent(coeff1,
                                            mfem.intArray(bdr_attr))
