from petram.mfem_config import use_parallel
import numpy as np


from petram.phys.coefficient import VCoeff
from petram.phys.vtable import VtableElement, Vtable

#from petram.phys.phys_model  import Phys, VectorPhysCoefficient
from petram.phys.em3d.em3d_base import EM3D_Bdry

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_E')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('E', VtableElement('E', type='complex',
                            guilabel='Electric field',
                            suffix=('x', 'y', 'z'),
                            default=np.array([0, 0, 0]),
                            tip="essential BC")),)

def bdry_constraints():
    return [EM3D_E]


class EM3D_E(EM3D_Bdry):
    has_essential = True
    vt = Vtable(data)

    def attribute_set(self, v):
        super(EM3D_E, self).attribute_set(v)

        # old file may not contain E_x_txt
        if not hasattr(self, "E_x_txt") and hasattr(self, "E_x"):
            dprint1("making adjustment for backward compatilibty")
            setattr(self, "E_x_txt", str(self.E_x))
            setattr(self, "E_y_txt", str(self.E_y))
            setattr(self, "E_z_txt", str(self.E_z))
        return v

    def get_essential_idx(self, kfes):
        if kfes == 0:
            return self._sel_index
        else:
            return []

    def apply_essential(self, engine, gf, real=False, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Apply Ess.(real)" + str(self._sel_index))
        else:
            dprint1("Apply Ess.(imag)" + str(self._sel_index))

        Exyz = self.vt.make_value_or_expression(self)

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        coeff1 = VCoeff(3, Exyz[0], ind_vars, l, g, return_complex=True)
        #f_name = self._make_f_name()
        # coeff1 = Et(3, f_name,  self.get_root_phys().ind_vars,
        #            self._local_ns, self._global_ns,
        #            real = real)
        coeff1 = self.restrict_coeff(coeff1, engine, vec=True)

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()
        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        coeff1 = coeff1.get_realimag_coefficient(real)
        gf.ProjectBdrCoefficientTangent(coeff1, mfem.intArray(bdr_attr))
