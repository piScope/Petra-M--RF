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

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_H')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('ht', VtableElement('ht', type='complex',
                             guilabel='H',
                             suffix=('r', 'phi', 'z'),
                             default=[0, 0, 0],
                             tip="boundary H field.")),)


class rHp(VectorPhysCoefficient):  # i \omega r Jext_t
    def __init__(self, *args, **kwargs):
        self.omega = kwargs.pop('omega', 1.0)
        super(rHp, self).__init__(*args, **kwargs)

    def Eval(self, V, T, ip):
        nor = mfem.Vector(2)
        mfem.CalcOrtho(T.Jacobian(), nor)
        tmp = nor.GetDataArray()
        tmp = np.array([tmp[0], 0, tmp[1]])
        self.nor = tmp/np.linalg.norm(tmp)

        return VectorPhysCoefficient.Eval(self, V, T, ip)

    def EvalValue(self, x):
        v = super(rHp, self).EvalValue(x)
        v = np.cross(self.nor, v)

        v = np.array((v[0], v[2]))
        v = -1j * self.omega * v * x[0]
        if self.real:
            return v.real
        else:
            return v.imag


class Ht(PhysCoefficient):  # i \omega Jext_phi
    def __init__(self, *args, **kwargs):
        self.omega = kwargs.pop('omega', 1.0)
        super(Ht, self).__init__(*args, **kwargs)

    def Eval(self, T, ip):
        nor = mfem.Vector(2)
        mfem.CalcOrtho(T.Jacobian(), nor)
        tmp = nor.GetDataArray()
        tmp = np.array([tmp[0], 0, tmp[1]])
        self.nor = tmp/np.linalg.norm(tmp)
        return PhysCoefficient.Eval(self, T, ip)

    def EvalValue(self, x):
        v = super(Ht, self).EvalValue(x)
        v = np.cross(self.nor, v)

        v = -1j * self.omega * v[1]

        if self.real:
            return v.real
        else:
            return v.imag


def bdry_constraints():
    return [EM2Da_H]


class EM2Da_H(EM2Da_Bdry):
    is_essential = False
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

            print(kfes, f_name)
            if kfes == 0:
                coeff1 = rHp(2, f_name[0],  self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real=real, omega=omega)
                self.add_integrator(engine, 'ht', coeff1,
                                    b.AddBoundaryIntegrator,
                                    mfem.VectorFEDomainLFIntegrator)
            else:
                coeff1 = Ht(f_name[0],  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real=real, omega=omega)
                self.add_integrator(engine, 'htor', coeff1,
                                    b.AddBoundaryIntegrator,
                                    mfem.DomainLFIntegrator)

        else:
            assert False, "should not come here"
