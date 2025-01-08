'''

   3D port boundary condition


    2016 5/20  first version only TE modes
'''
from petram.phys.vtable import VtableElement, Vtable
from petram.mfem_config import use_parallel
from petram.phys.phys_const import epsilon0, mu0
from mfem.common.mpi_debug import nicePrint
import sys
import numpy as np

from petram.model import Bdry
from petram.phys.phys_model import Phys
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

from petram.helper.geom import connect_pairs2
from petram.helper.geom import find_circle_center_radius

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Port')


if use_parallel:
    import mfem.par as mfem
    '''
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   '''
else:
    import mfem.ser as mfem

'''
  TE Mode (Ephi only)
    Expression are based on Microwave Engineering p122 - p123.
    Note that it consists from two terms
       1)  \int dS W \dot n \times iwH (VectorFETangentIntegrator does this)
       2)  an weighting to evaulate mode amplutude from the E field
           on a boundary


'''

data = (('inc_amp', VtableElement('inc_amp', type='complex',
                                  guilabel='incoming amp',
                                  default=1.0,
                                  tip="amplitude of incoming wave")),
        ('inc_phase', VtableElement('inc_phase', type='float',
                                    guilabel='incoming phase (deg)',
                                    default=0.0,
                                    tip="phase of incoming wave")),
        ('epsilonr', VtableElement('epsilonr', type='complex',
                                   guilabel='epsilonr',
                                   default=1.0,
                                   tip="relative permittivity")),
        ('mur', VtableElement('mur', type='complex',
                              guilabel='mur',
                              default=1.0,
                              tip="relative permeability")),)

'''
   E_(name)_rz  :
   E_(name)_phi :
   H_(name)_rz  : Vector (Hphi, Hphi)
   H_(name)_phi : Scalar (nr Hz - nz Hr)

   n x H  = (-nz * Hphi, nr*Hz - nz*Hr, nr * Hphi)
'''
'''
   rectangular wg TE
'''


class E_TE_phi(mfem.PyCoefficient):
    def __init__(self, bdry, real=True, amp=1.0, eps=1.0, mur=1.0):

        self.real = real
        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.a, self.c = bdry.a, bdry.c
        self.a_vec = bdry.a_vec
        self.m, self.n = bdry.mn[0], bdry.mn[1]
        mfem.PyCoefficient.__init__(self)

    def EvalValue(self, x):
        p = np.array(x)
        # x, y is positive (first quadrant)
        xx = np.sqrt(np.sum((p - self.c)**2)) / self.a  # 0 < x < 1
        Ephi = self.m / self.a * \
            np.sin(self.m * np.pi * xx) * (x[0])  # Ephi = -rho Ephi
        if self.real:
            return -Ephi.real
        else:
            return -Ephi.imag


class H_TE_rz(mfem.VectorPyCoefficient):
    def __init__(
            self,
            sdim,
            phase,
            bdry,
            real=True,
            amp=1.0,
            eps=1.0,
            mur=1.0):
        mfem.VectorPyCoefficient.__init__(self, sdim)

        self.real = real
        self.phase = phase  # phase !=0 for incoming wave

        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.a, self.c = bdry.a, bdry.c
        self.a_vec = bdry.a_vec
        self.m, self.n = bdry.mn[0], bdry.mn[1]

        k = omega * np.sqrt(eps * epsilon0 * mur * mu0)
        kc = np.abs(bdry.mn[0] * np.pi / bdry.a)
        if kc > k:
            raise ValueError('Mode does not propagate')
        beta = np.sqrt(k**2 - kc**2)
        dprint1("propagation constant:" + str(beta))

        AA = omega * mur * mu0 * np.pi / kc / kc
        self.AA = omega * beta * np.pi / kc / kc / AA * amp

    def EvalValue(self, x):
        p = np.array(x)
        # x, y is positive (first quadrant)
        xx = np.sqrt(np.sum((p - self.c)**2)) / self.a  # 0 < x < 1
        H = 1j * self.AA * self.m / self.a * np.sin(self.m * np.pi * xx)
        H = -H * np.exp(1j * self.phase / 180. * np.pi) * self.a_vec * x[0]
        H = np.array([H[0], 0])
        # return np.array([1,1.])
        if self.real:
            return H.real
        else:
            return H.imag


class H_TE_phi(mfem.PyCoefficient):
    def __init__(self, phase, bdry, real=True, amp=1.0, eps=1.0, mur=1.0):
        mfem.PyCoefficient.__init__(self)

        self.real = real
        self.phase = phase  # phase !=0 for incoming wave

        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.a, self.c = bdry.a, bdry.c
        self.a_vec = bdry.a_vec
        self.m, self.n = bdry.mn[0], bdry.mn[1]

        k = omega * np.sqrt(eps * epsilon0 * mur * mu0)
        kc = np.abs(bdry.mn[0] * np.pi / bdry.a)
        if kc > k:
            raise ValueError('Mode does not propagate')
        beta = np.sqrt(k**2 - kc**2)
        dprint1("propagation constant:" + str(beta))

        AA = omega * mur * mu0 * np.pi / kc / kc * amp
        self.AA = omega * beta * np.pi / kc / kc / AA
        self.beta = beta
        self.fac = 1 / mu0 / mur * amp
        '''
        AA = beta/amp/mu
       '''

    def EvalValue(self, x):
        p = np.array(x)
        # x, y is positive (first quadrant)
        xx = np.sqrt(np.sum((p - self.c)**2)) / self.a  # 0 < x < 1
        #H = 1j*self.AA* self.m/self.a*np.sin(self.m*np.pi*xx)
        H = (-1j * self.beta - 1 / x[0]) * self.fac * \
            self.m / self.a * np.sin(self.m * np.pi * xx)
        H = H * np.exp(1j * self.phase / 180. * np.pi)
        if self.real:
            return H.real
        else:
            return H.imag


H_TE_rz = None
E_TE_rz = None


# Ephi mode
class E_Ephi_rz(mfem.VectorPyCoefficient):
    pass


class E_Ephi_phi(mfem.PyCoefficient):
    pass


class H_Ephi_rz(mfem.VectorPyCoefficient):
    pass


class H_Ephi_phi(mfem.PyCoefficient):
    pass

def bdry_constraints():
   return [EM2Da_Port]

class EM2Da_Port(EM2Da_Bdry):
    extra_diagnostic_print = True
    vt = Vtable(data)

    def __init__(self, mode='TE', mn='0,1', inc_amp='1',
                 inc_phase='0', port_idx=1):
        super(EM2Da_Port, self).__init__(mode=mode,
                                         mn=mn,
                                         inc_amp=inc_amp,
                                         inc_phase=inc_phase,
                                         port_idx=port_idx)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM2Da_Port, self).attribute_set(v)
        v['port_idx'] = 1
        v['mode'] = 'TE'
        v['mn'] = [1, 0]
        v['epsilonr'] = 1.0
        v['mur'] = 1.0
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['isTimeDependent_RHS'] = True
        self.vt.attribute_set(v)
        return v

    def panel1_param(self):
        return ([["port id", str(self.port_idx), 0, {}],
                 ["mode (n=0 only)", self.mode, 4, {"readonly": True,
                                                    "choices": ["TE", ]}],
                 ["m/n", ','.join(str(x) for x in self.mn), 0, {}], ] +
                self.vt.panel_param(self))

    def get_panel1_value(self):
        return ([str(self.port_idx),
                 self.mode, ','.join(str(x) for x in self.mn)] +
                self.vt.get_panel_value(self))

    def import_panel1_value(self, v):
        self.port_idx = v[0]
        self.mode = v[1]
        self.mn = [int(x) for x in v[2].split(',')]
        self.vt.import_panel_value(self, v[3:])

    def panel4_param(self):
        ll = super(EM2Da_Port, self).panel4_param()
        ll.append(['Varying (in time/for loop) RHS', False, 3, {"text": ""}])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        super(EM2Da_Port, self).import_panel4_value(value[:-1])
        self.isTimeDependent_RHS = value[-1]

    def get_panel4_value(self):
        value = super(EM2Da_Port, self).get_panel4_value()
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
            self.inc_amp = self.eval_phys_expr(
                str(self.inc_amp_txt), 'inc_amp')[0]
            self.inc_phase = self.eval_phys_expr(
                str(self.inc_phase_txt), 'inc_phase', chk_float=True)[0]
        except BaseException:
            raise ValueError("Cannot evaluate amplitude/phase to float number")

    def preprocess_params(self, engine):
        # find normal (outward) vector...
        mesh = engine.get_emesh(mm=self)
        fespace = engine.fespaces[self.get_root_phys().dep_vars[0]]
        nbe = mesh.GetNBE()
        ibe = np.array([i for i in range(nbe)
                        if mesh.GetBdrElement(i).GetAttribute() ==
                        self._sel_index[0]])
        dprint1("idb", ibe)
        el = mesh.GetBdrElement(ibe[0])
        Tr = fespace.GetBdrElementTransformation(ibe[0])
        rules = mfem.IntegrationRules()
        ir = rules.Get(el.GetGeometryType(), 1)
        Tr.SetIntPoint(ir.IntPoint(0))
        nor = mfem.Vector(2)
        mfem.CalcOrtho(Tr.Jacobian(), nor)

        self.norm = nor.GetDataArray().copy()
        self.norm = self.norm / np.sqrt(np.sum(self.norm**2))

        dprint1("Normal Vector " + list(self.norm).__repr__())

        # find rectangular shape

        edges = np.array([mesh.GetBdrElementEdges(i)[0]
                          for i in ibe]).flatten()
        d = {}
        for x in edges:
            d[x] = x in d
        edges = [x for x in d.keys() if not d[x]]
        ivert = [mesh.GetEdgeVertices(x) for x in edges]
        ivert = connect_pairs2(ivert)[0]

        vv = np.vstack([mesh.GetVertexArray(i) for i in ivert])
        self.ctr = (vv[0] + vv[-1]) / 2.0
        dprint1("Center " + list(self.ctr).__repr__())

    # rectangular port
        self.a_vec = (vv[0] - vv[-1])
        self.a = np.sqrt(np.sum(self.a_vec**2))
        self.a_vec /= self.a
        self.c = vv[-1]
        dprint1("Cornor " + self.c.__repr__())
        dprint1("Edge  " + self.a.__repr__())
        dprint1("Edge Vec." + list(self.a_vec).__repr__())

        Erz, Ephi = self.get_e_coeff_cls()
        Hrz, Hphi = self.get_h_coeff_cls()

        if self.mode == 'TE':
            dprint1("E field pattern")
            c1 = Ephi(self, real=True)
            c2 = Hphi(0.0, self, real=False)
            for p in vv:
                dprint1(p.__repr__() + ' : ' + c1.EvalValue(p).__repr__())
            dprint1("H field pattern")
            for p in vv:
                dprint1(p.__repr__() + ' : ' + c2.EvalValue(p).__repr__())

    def get_h_coeff_cls(self):
        m = sys.modules[__name__]
        c1 = getattr(m, 'H_' + self.mode + '_rz')
        c2 = getattr(m, 'H_' + self.mode + '_phi')
        return c1, c2

    def get_e_coeff_cls(self):
        m = sys.modules[__name__]
        c1 = getattr(m, 'E_' + self.mode + '_rz')
        c2 = getattr(m, 'E_' + self.mode + '_phi')
        return c1, c2

    def has_lf_contribution(self, kfes):
        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)
        if kfes == 1:
            if self.mode == 'TE':
                return inc_amp != 0
            elif self.mode == 'Ephi':
                return inc_amp != 0
        elif kfes == 0:
            if self.mode == 'TE':
                return inc_amp != 0
            elif self.mode == 'Ephi':
                return False
        else:
            return False

    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        if real:
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        Hrz, Hphi = self.get_h_coeff_cls()
        inc_wave = inc_amp * np.exp(1j * inc_phase / 180. * np.pi)
        # inc_wave = inc_amp * np.exp(-1j*inc_phase/180.*np.pi)  # original...
        #assert False, "when you use for the first time, the sign must be checked...)"
        phase = np.angle(inc_wave) * 180 / np.pi
        amp = np.abs(inc_wave)

#        if (kfes == 0 and self.mode == 'TE'):  # E~phi B ~rz
#            coeff = Hrz(2, phase, self, real = real, amp = amp)
#            self.add_integrator(engine, 'inc_amp', coeff,
#                                b.AddBoundaryIntegrator,
#                                mfem.VectorFEDomainLFIntegrator)
#                                mfem.VectorFEBoundaryTangentLFIntegrator)
        if (kfes == 1 and self.mode == 'TE'):  # E~phi B ~rz
            coeff = Hphi(phase, self, real=real, amp=amp, eps=eps, mur=mur)
            self.add_integrator(engine, 'inc_amp', coeff,
                                b.AddBoundaryIntegrator,
                                mfem.BoundaryLFIntegrator)

        else:
            pass

    def has_extra_DoF(self, kfes):
        if self.mode == 'TE' and kfes == 1:
            return True
        elif kfes == 0:
            if self.mode == 'Ephi':
                return True
        else:
            return False

    def get_extra_NDoF(self):
        return 1

    # def get_probe(self):
    #    return self.get_root_phys().dep_vars[0]+"_port_"+str(self.port_idx)

    def postprocess_extra(self, sol, flag, sol_extra):
        name = self.name() + '_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()

    def add_extra_contribution(self, engine, **kwargs):
        from mfem.common.chypre import LF2PyVec

        kfes = kwargs.pop('kfes', 0)
        dprint1("Add Extra contribution" + str(self._sel_index))

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        Erz, Ephi = self.get_e_coeff_cls()
        Hrz, Hphi = self.get_h_coeff_cls()
        fes = engine.get_fes(self.get_root_phys(), kfes)

        '''
        if (kfes == 0 and self.mode == 'TE'):
           lf1 = engine.new_lf(fes)
           Ht = Hrz(2, 0.0, self, real = True)
           Ht = self.restrict_coeff(Ht, engine, vec = True)
           intg =  mfem.VectorFEDomainLFIntegrator(Ht)
           #intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
           lf1.AddBoundaryIntegrator(intg)
           lf1.Assemble()
           lf1i = engine.new_lf(fes)
           Ht = Hrz(2, 0.0, self, real = False)
           Ht = self.restrict_coeff(Ht, engine, vec = True)
           intg =  mfem.VectorFEDomainLFIntegrator(Ht)
           #intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
           lf1i.AddBoundaryIntegrator(intg)
           lf1i.Assemble()

           from mfem.common.chypre import LF2PyVec
           v1 = LF2PyVec(lf1, lf1i)
           v1 *= -1
           # output formats of InnerProduct
           # are slightly different in parallel and serial
           # in serial numpy returns (1,1) array, while in parallel
           # MFEM returns a number. np.sum takes care of this.
           return (v1, None, None, None, False)
        '''
        if (kfes == 1 and self.mode == 'TE'):

            lf1 = engine.new_lf(fes)
            Ht = Hphi(0.0, self, real=True, eps=eps, mur=mur)
            Ht = self.restrict_coeff(Ht, engine)
            intg = mfem.BoundaryLFIntegrator(Ht)
            lf1.AddBoundaryIntegrator(intg)
            lf1.Assemble()

            lf1i = engine.new_lf(fes)
            Ht = Hphi(0.0, self, real=False, eps=eps, mur=mur)
            Ht = self.restrict_coeff(Ht, engine)
            intg = mfem.BoundaryLFIntegrator(Ht)
            lf1i.AddBoundaryIntegrator(intg)
            lf1i.Assemble()

            from mfem.common.chypre import LF2PyVec, PyVec2PyMat, Array2PyVec, IdentityPyMat

            v1 = LF2PyVec(lf1, lf1i)
            #v1 *= -1

            lf2 = engine.new_lf(fes)
            Et = Ephi(self, real=True, eps=eps, mur=mur)
            Et = self.restrict_coeff(Et, engine)
            intg = mfem.DomainLFIntegrator(Et)
            lf2.AddBoundaryIntegrator(intg)
            lf2.Assemble()

            x = engine.new_gf(fes)
            x.Assign(0.0)
            arr = self.get_restriction_array(engine)
            x.ProjectBdrCoefficient(Et, arr)

            weight = mfem.InnerProduct(engine.x2X(x), engine.b2B(lf2))
            v2 = LF2PyVec(lf2, None, horizontal=True)
            v2 *= -1 / weight / 2.0

            #x  = LF2PyVec(x, None)
            # output formats of InnerProduct
            # are slightly different in parallel and serial
            # in serial numpy returns (1,1) array, while in parallel
            # MFEM returns a number. np.sum takes care of this.
            #tmp = np.sum(v2.dot(x))
            #v2 *= -1/tmp/2.

            t4 = np.array([[inc_amp * np.exp(1j * inc_phase / 180. * np.pi)]])

            # convert to a matrix form
            v1 = PyVec2PyMat(v1)
            v2 = PyVec2PyMat(v2.transpose())
            t4 = Array2PyVec(t4)
            t3 = IdentityPyMat(1, diag=-1)

            v2 = v2.transpose()

            # return (None, v2, t3, t4, True)
            return (v1, v2, t3, t4, True)
