'''
   Essential BC
'''
from petram.phys.numba_coefficient import func_to_numba_coeff_scalar
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.coefficient import VCoeff
from petram.mfem_config import use_parallel
import numpy as np

from petram.model import Bdry
#
#  symmatry axis
#
#    m = 0   : Ephi = 0, Er = 0 , \partial_r Ez = 0
#    m = +/-  : Ez = 0, Ephi + iEr = 0
#    m >= 2 or ==-2  : Ez = 0 and Ephi = 0
#

from petram.phys.phys_model import Phys, VectorPhysCoefficient, PhysCoefficient
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_E')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem


data = (('t_mode', VtableElement('t_mode', type='int',
                                 guilabel = 'm',
                                 default = 0.0, 
                                 tip = "mode number" )),)

def bdry_constraints():
    return [EM2Da_Axis]


class EM2Da_Axis(EM2Da_Bdry):
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

        tmode = self.vt.make_value_or_expression(self)[0]

        ind_vars = self.get_root_phys().ind_vars
        l = self._local_ns
        g = self._global_ns

        mesh = engine.get_mesh(mm=self)
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()

        for idx in self._sel_index:
            bdr_attr[idx-1] = 1

        if abs(tmode) == 0:
            # Ephi = 0 as essential
            if kfes == 1:
                coeff1 = mfem.ConstantCoefficient(0.0)
                gf.ProjectBdrCoefficient(coeff1,
                                         mfem.intArray(bdr_attr))
        elif abs(tmode) == 1:
            # Ez = 0 as essential            
            if kfes == 0:
                coeff1 = mfem.VectorArrayCoefficient(2)
                coeff1.Set(0, mfem.ConstantCoefficient(0.0))
                coeff1.Set(1, mfem.ConstantCoefficient(0.0))            
                gf.ProjectBdrCoefficientTangent(coeff1,
                                            mfem.intArray(bdr_attr))
            assert False, "this Bdr is not implemented completely"
        elif abs(tmode) >= 2:
            # Ephi = Ez = 0 as essential                        
            if kfes == 0:
                coeff1 = mfem.VectorArrayCoefficient(2)
                coeff1.Set(0, mfem.ConstantCoefficient(0.0))
                coeff1.Set(1, mfem.ConstantCoefficient(0.0))            
                gf.ProjectBdrCoefficientTangent(coeff1,
                                            mfem.intArray(bdr_attr))
            elif kfes == 1:
                coeff1 = mfem.ConstantCoefficient(0.0)
                gf.ProjectBdrCoefficient(coeff1,
                                         mfem.intArray(bdr_attr))
            
