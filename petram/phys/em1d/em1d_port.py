'''
   1D port boundary condition
'''
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
from petram.phys.phys_const import mu0, epsilon0

from mfem.common.mpi_debug import nicePrint
import sys
import numpy as np

from petram.model import Bdry
from petram.phys.phys_model import Phys
from petram.phys.em1d.em1d_base import EM1D_Bdry, EM1D_Domain

from petram.helper.geom import connect_pairs2
from petram.helper.geom import find_circle_center_radius

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM1D_Port')


if use_parallel:
    import mfem.par as mfem
    '''
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   '''
else:
    import mfem.ser as mfem


data = (('inc_amp', VtableElement('inc_amp', type='complex',
                                  guilabel='inc. amp.',
                                  default=(1.0, 0.0),
                                  suffix=('y', 'z'),
                                  no_func=True,
                                  tip="amplitude of incoming wave")),
        ('inc_phase', VtableElement('inc_phase', type='float',
                                    guilabel='inc. phase (deg)',
                                    default=0.0,
                                    no_func=True,
                                    tip="phase of incoming wave")),
        ('epsilonr', VtableElement('epsilonr', type='complex',
                                   guilabel='epsilonr',
                                   default=1.0,
                                   no_func=True,
                                   tip="relative permittivity")),
        ('mur', VtableElement('mur', type='complex',
                              guilabel='mur',
                              default=1.0,
                              no_func=True,
                              tip="relative permeability")),
        ('ky', VtableElement('ky', type='float',
                             guilabel='ky',
                             default=0.,
                             no_func=True,
                             tip="wave number` in the y direction")),
        ('kz', VtableElement('kz', type='float',
                             guilabel='kz',
                             default=0.0,
                             no_func=True,
                             tip="wave number in the z direction")),)


def bdry_constraints():
    return [EM1D_Port]


class E_port(mfem.PyCoefficient):
    def __init__(self, bdry, real=True, amp=(1.0, 0.0),  eps=1.0,
                 mur=1.0, ky=0.0, kz=0.0, direction="y", normalize=False):

        mfem.PyCoefficient.__init__(self)
        self.real = real
        freq, omega = bdry.get_root_phys().get_freq_omega()

        k = omega*np.sqrt(eps*epsilon0 * mur*mu0)
        kc2 = k**2 - ky**2 - kz**2
        if kc2 < 0:
            kc2 = complex(kc2)
            #raise ValueError('Mode does not propagate')
        beta = np.sqrt(kc2)
        dprint1("propagation constant:" + str(beta))
        Ey, Ez = amp
        Ex = -(Ey*ky + Ez*kz)/beta
        if normalize:
            E_norm = np.sqrt(Ex*np.conj(Ex) + Ey*np.conj(Ey) + Ez*np.conj(Ez))
            Ex = Ex/E_norm
            Ey = Ey/E_norm
            Ez = Ez/E_norm
        if direction == "y":
            self.ret = Ey
        else:
            self.ret = Ez
        self.ret = complex(self.ret)

    def EvalValue(self, x):
        if self.real:
            return self.ret.real
        else:
            return self.ret.imag


class jwH_port(mfem.PyCoefficient):
    def __init__(self, bdry, real=True, amp=(1.0, 0.0),  eps=1.0,
                 mur=1.0, ky=0.0, kz=0.0, direction="y", normalize=False):
        mfem.PyCoefficient.__init__(self)
        self.real = real
        freq, omega = bdry.get_root_phys().get_freq_omega()

        k = omega*np.sqrt(eps*epsilon0 * mur*mu0)
        kc2 = k**2 - ky**2 - kz**2

        if kc2 < 0:
            dprint1('Mode does not propagate !!!')
            kc2 = complex(kc2)
            #raise ValueError('Mode does not propagate')

        norm = -bdry.norm  # norm is INWARD propagation
        beta = np.sqrt(kc2) * norm
        dprint1("propagation constant:" + str(beta))
        Ey, Ez = amp
        Ex = -(Ey*ky + Ez*kz)/beta

        if normalize:
            E_norm = np.sqrt(Ex*np.conj(Ex) + Ey*np.conj(Ey) + Ez*np.conj(Ez))
            Ex = Ex/E_norm
            Ey = Ey/E_norm
            Ez = Ez/E_norm
        Hy = 1j*(-beta*Ez + kz*Ex)/mu0/mur
        Hz = 1j*(beta*Ey - ky*Ex)/mu0/mur

        # return n x H (n is outward vector
        if direction == "y":
            self.ret = -Hz * (-bdry.norm)
        else:
            self.ret = Hy * (-bdry.norm)
        self.ret = complex(self.ret)

    def EvalValue(self, x):
        if self.real:
            return self.ret.real
        else:
            return self.ret.imag


class EM1D_Port(EM1D_Bdry):
    extra_diagnostic_print = True
    vt = Vtable(data)

    def __init__(self, inc_amp='1, 0', inc_phase='0', port_idx=1):
        super(EM1D_Port, self).__init__(inc_amp=inc_amp,
                                        inc_phase=inc_phase,
                                        port_idx=port_idx)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM1D_Port, self).attribute_set(v)
        v['port_idx'] = 1
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['isTimeDependent_RHS'] = True
        self.vt.attribute_set(v)
        return v

    def panel1_param(self):
        return ([["port id", str(self.port_idx), 0, {}], ] +
                self.vt.panel_param(self))

    def get_panel1_value(self):
        return ([str(self.port_idx), ] +
                self.vt.get_panel_value(self))

    def import_panel1_value(self, v):
        self.port_idx = v[0]
        self.vt.import_panel_value(self, v[1:])

    def panel4_param(self):
        ll = super(EM1D_Port, self).panel4_param()
        ll.append(['Varying (in time/for loop) RHS', False, 3, {"text": ""}])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        super(EM1D_Port, self).import_panel4_value(value[:-1])
        self.isTimeDependent_RHS = value[-1]

    def get_panel4_value(self):
        value = super(EM1D_Port, self).get_panel4_value()
        value.append(self.isTimeDependent_RHS)
        return value

    def verify_setting(self):
        if self.isTimeDependent_RHS:
            flag = True
        else:
            flag = False
        return flag, 'Varying RHS is not set', 'This potntially causes an error with PortScan. Set it Time/NL Dep. panel '

    def update_param(self):
        self.update_inc_amp_phase()

    def update_inc_amp_phase(self):
        '''
        set parameter
        '''
        try:
            self.vt.preprocess_params(self)
            inc_amp, inc_phase, eps, mur, ky, kz = self.vt.make_value_or_expression(
                self)
            self.inc_amp = inc_amp
            self.inc_phase = inc_phase
        except:
            raise ValueError("Cannot evaluate amplitude/phase to float number")

    def preprocess_params(self, engine):
        # find normal (outward) vector...
        mesh = engine.get_emesh(mm=self)
        fespace = engine.fespaces[self.get_root_phys().dep_vars[1]]

        nbe = mesh.GetNBE()

        BdrPtx = []
        for i in range(nbe):
            iv = mesh.GetBdrElementVertices(i)
            BdrPtx.append(mesh.GetVertexArray(iv[0])[0])

        ibe = np.array([i for i in range(nbe)
                        if mesh.GetBdrElement(i).GetAttribute() ==
                        self._sel_index[0]])
        iv = mesh.GetBdrElementVertices(ibe[0])
        ptx = mesh.GetVertexArray(iv[0])[0]
        if ptx == BdrPtx[-1]:
            self.norm = 1.
        else:
            self.norm = -1.

    def has_lf_contribution(self, kfes):
        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur, ky, kz = self.vt.make_value_or_expression(
            self)
        if kfes == 1:
            return inc_amp[0] != 0
        elif kfes == 2:
            return inc_amp[1] != 0
        else:
            return False

    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        if kfes == 0:
            return

        txt = ["Ex in ", "Ey in ", "Ez in "][kfes]
        if real:
            dprint1("Add LF contribution(real) " + txt + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + txt + str(self._sel_index))

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur, ky, kz = self.vt.make_value_or_expression(
            self)

        inc_wave = np.array(inc_amp) * np.exp(1j*inc_phase/180.*np.pi)
        d = "y" if kfes == 1 else "z"

        # note for the right hand side, we multiple -1 to ampulitude

        coeff = jwH_port(self, real=real, amp=inc_wave,  eps=eps,
                         mur=mur, ky=ky, kz=kz,  direction=d)
        self.add_integrator(engine, 'inc_amp', coeff,
                            b.AddBoundaryIntegrator,
                            mfem.BoundaryLFIntegrator)

    def has_extra_DoF(self, kfes):
        if kfes == 0:
            return False
        else:
            return True

    def get_extra_NDoF(self):
        return 1

    def extra_DoF_name2(self, kfes=0):
        '''
        default DoF name
        '''
        if kfes == 0:
            assert False, "should not come here"
        elif kfes == 1:
            return self.get_root_phys().dep_vars[1]+"_port"+str(self.port_idx)
        elif kfes == 2:
            return self.get_root_phys().dep_vars[2]+"_port"+str(self.port_idx)

    def get_probes(self):
        return [self.get_root_phys().dep_vars[1]+"_port"+str(self.port_idx),
                self.get_root_phys().dep_vars[2]+"_port"+str(self.port_idx), ]

    def postprocess_extra(self, sol, flag, sol_extra):
        assert False, "Is it used?"
        name = self.name()+'_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()

    def add_extra_contribution(self, engine, **kwargs):

        from mfem.common.chypre import LF2PyVec

        kfes = kwargs.pop('kfes', 0)
        if kfes == 0:
            return
        dprint1("Add Extra contribution" + str(self._sel_index))

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur, ky, kz = self.vt.make_value_or_expression(
            self)

        fes = engine.get_fes(self.get_root_phys(), kfes)
        d = "y" if kfes == 1 else "z"
        inc_amp0 = (1, 0) if kfes == 1 else (0, 1)
        lf1 = engine.new_lf(fes)
        Ht = jwH_port(self, real=True, amp=inc_amp0,  eps=eps,
                      mur=mur, ky=ky, kz=kz,  direction=d)  # , normalize=True)
        Ht = self.restrict_coeff(Ht, engine)
        intg = mfem.BoundaryLFIntegrator(Ht)
        lf1.AddBoundaryIntegrator(intg)
        lf1.Assemble()

        lf1i = engine.new_lf(fes)
        Ht = jwH_port(self, real=False, amp=inc_amp0,  eps=eps,
                      mur=mur, ky=ky, kz=kz,  direction=d)  # , normalize=True)
        Ht = self.restrict_coeff(Ht, engine)
        intg = mfem.BoundaryLFIntegrator(Ht)
        lf1i.AddBoundaryIntegrator(intg)
        lf1i.Assemble()

        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, Array2PyVec, IdentityPyMat

        v1 = LF2PyVec(lf1, lf1i)
        #v1 *= -1

        lf2 = engine.new_lf(fes)
        Et = E_port(self, real=True, amp=inc_amp0,  eps=eps,
                    mur=mur, ky=ky, kz=kz,  direction=d)  # , normalize=True)

        Et = self.restrict_coeff(Et, engine)
        intg = mfem.DomainLFIntegrator(Et)
        lf2.AddBoundaryIntegrator(intg)
        lf2.Assemble()

        x = engine.new_gf(fes)
        x.Assign(0.0)
        arr = self.get_restriction_array(engine)
        x.ProjectBdrCoefficient(Et,  arr)

        t4 = np.array([[inc_amp[kfes-1]*np.exp(1j*inc_phase/180.*np.pi)]])

        weight = mfem.InnerProduct(engine.x2X(x), engine.b2B(lf2))

        v2 = LF2PyVec(lf2, None, horizontal=True)
        x = LF2PyVec(x, None)

        v2 *= 1/weight

        # convert to a matrix form
        v1 = PyVec2PyMat(v1)
        v2 = PyVec2PyMat(v2.transpose())
        t4 = Array2PyVec(t4)
        t3 = IdentityPyMat(1, diag=-1)

        v2 = v2.transpose()

        # return (None, v2, t3, t4, True)
        return (v1, v2, t3, t4, True)
