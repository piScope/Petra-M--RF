'''

   EM2Da port boundary condition


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
    from mfem.common.mpi_debug import nicePrint
    from petram.helper.mpi_recipes import *    
else:
    import mfem.ser as mfem

'''
  TE Mode (Ephi only)
    Expression are based on Microwave Engineering p122 - p123.
    Note that it consists from two terms
       1)  \int dS W \dot n \times iwH (VectorFETangentIntegrator does this)
       2)  an weighting to evaulate mode amplutude from the E field
           on a boundary

  Coax Mode:
    Coaxial port on horizontal boundary

  Circular Mode:
    Circular port centered at axis. on horizontal boundary

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
        cnorm = bdry.get_root_phys().get_coeff_norm()

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
        self.AA = omega * beta * np.pi / kc / kc / AA * amp * cnorm

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
        cnorm = bdry.get_root_phys().get_coeff_norm()

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
        self.AA = omega * beta * np.pi / kc / kc / AA * cnorm
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

def get_rz_cls(basecls, mode="H"):
    # use 3D version of port mode behind. Then give r and z components.
    if mode == "H":
        class cls_rz(mfem.VectorPyCoefficient):
            def __init__(self, *args, **kwags):
                self.base = basecls(3, *args[1:], **kwags) 
                bdr = args[1]
                self.norm = np.array((bdr.norm[0], 0.0, bdr.norm[1]))
                mfem.VectorPyCoefficient.__init__(self, args[0])

            def EvalValue(self, x):
                xx = np.array((x[0], 0, x[1]))
                val = self.base.EvalValue(xx)
                print("r, z component", self.norm, x[0], val[0], val[2])
                val = np.cross(self.norm, val)
                return x[0]*val[0], x[0]*val[2]
                #return val[0], val[2]
    else:
        class cls_rz(mfem.VectorPyCoefficient):
            def __init__(self, *args, **kwags):
                self.base = basecls(3, *args[1:], **kwags) 
                mfem.VectorPyCoefficient.__init__(self, args[0])

            def EvalValue(self, x):
                xx = np.array((x[0], 0, x[1]))
                val = self.base.EvalValue(xx)
                print("r, z component", x[0], val[0], val[2])
                return val[0], val[2]
        
    return cls_rz

def get_phi_cls(basecls, mode="H"):
    # use 3D version of port mode behind. Then give r * Aphi
    if mode == "H":    
            class cls_phi(mfem.PyCoefficient):
                def __init__(self, *args, **kwags):
                    self.base = basecls(3, *args, **kwags) 
                    bdr = args[0]
                    self.norm = np.array((bdr.norm[0], 0.0, bdr.norm[1]))
                    mfem.PyCoefficient.__init__(self)            

                def EvalValue(self, x):
                    xx = np.array((x[0], 0, x[1]))
                    val = self.base.EvalValue(xx)
                    val = np.cross(self.norm, val)                    
                    print("phi component", x[0], val[1])
                    return val[1]
    else:
        class cls_phi(mfem.PyCoefficient):
            def __init__(self, *args, **kwags):
                self.base = basecls(3, *args, **kwags) 
                mfem.PyCoefficient.__init__(self)            

            def EvalValue(self, x):
                xx = np.array((x[0], 0, x[1]))
                val = self.base.EvalValue(xx)
                print("phi component", x[0], x[0]*val[1])
                return x[0]*val[1]

    return cls_phi

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
        v['fixed_total_field'] = False        
        self.vt.attribute_set(v)
        return v

    def panel1_param(self):
        from petram.phys.common.rf_portmode import port_modes        

        return ([["port id", str(self.port_idx), 0, {}],
                 ["mode (n=0 only)", self.mode, 4, {"readonly": True,
                                                    "choices": port_modes}],
                 ["m/n", ','.join(str(x) for x in self.mn), 0, {}],
                 ["fixed excitaiton", True, 3, {"text": ""}],] + 
                self.vt.panel_param(self))

    def get_panel1_value(self):
        return ([str(self.port_idx),
                 self.mode, ','.join(str(x) for x in self.mn),
                 self.fixed_total_field, ] +
                self.vt.get_panel_value(self))

    def import_panel1_value(self, v):
        self.port_idx = v[0]
        self.mode = v[1]
        self.mn = [int(x) for x in v[2].split(',')]
        self.fixed_total_field = v[3]
        self.vt.import_panel_value(self, v[4:])

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

        rmin = np.min(vv[:, 0])
        rmax = np.max(vv[:, 0])        
        rc = (rmax + rmin)/2.0        
        zmax = np.min(vv[:, 1])                
        zmin = np.max(vv[:, 1])                
        zc = (zmax + zmin)/2.0

        dprint1("Center: ",  list(self.ctr).__repr__())
        dprint1("rmin/rmax: ",  rmin, rmax)

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

        if str(self.mode).upper().strip() in ['TE', 'TM', 'TEM']:
            self.a = geom_data["a"]
            self.a_vec = geom_data["a_vec"]
            self.b = geom_data["b"]
            self.b_vec = geom_data["b_vec"]
            self.c = geom_data["c"]
            dprint1("Long Edge  " + self.a.__repr__())
            dprint1("Long Edge Vec." + list(self.a_vec).__repr__())
            dprint1("Short Edge  " + self.b.__repr__())
            dprint1("Short Edge Vec." + list(self.b_vec).__repr__())

        elif self.mode == 'Coax(TEM)':
            # coax port is 
            assert zc == zmin, "Port should be horizontal edge"
            assert zc == zmax, "Port should be horizontal edge"

            self.a = rmin
            self.b = rmax
            self.ctr = np.array((0., 0., zc))
            self.ax1 = np.array((1., 0., 0.))
            self.ax2 = np.array((0., 1., 0.))                               
            
        elif self.mode == 'Circular(TE)':
            self.a = rmax
            self.ctr = np.array((0., 0., zc))
            self.ax1 = np.array((1., 0., 0.))
            self.ax2 = np.array((0., 1., 0.))                               
        else:
            assert False, "unknown mode"                                


    def get_h_coeff_cls(self):
        from petram.phys.common.rf_portmode import get_portmode_coeff_cls

        _Et, jwHt = get_portmode_coeff_cls(self.mode)

        cls_rz = get_rz_cls(jwHt, "H")
        cls_phi = get_phi_cls(jwHt, "H")
            
        return cls_rz, cls_phi

    def get_e_coeff_cls(self):
        from petram.phys.common.rf_portmode import get_portmode_coeff_cls

        Et, _jwHt = get_portmode_coeff_cls(self.mode)

        cls_rz = get_rz_cls(Et, "E")
        cls_phi = get_phi_cls(Et, "E")
        
        return cls_rz, cls_phi

    def has_lf_contribution(self, kfes):
        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        assert self.mode != 'Ephi', "Ephi mode is not supported anymore"

        return inc_amp != 0

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
        
        cnorm = self.get_root_phys().get_coeff_norm()

        if kfes == 0:
            coeff = Hrz(2, self, real=real, eps=eps, mur=mur, amp=amp, phase=phase, 
                        cnorm=cnorm, m=self.mn[0], n=self.mn[1])
            self.add_integrator(engine, 'inc_amp', coeff,
                                b.AddBoundaryIntegrator,
                                mfem.VectorFEDomainLFIntegrator)
                                #mfem.VectorFEBoundaryTangentLFIntegrator)
        elif kfes == 1:
            coeff = Hphi(self, real=real, eps=eps, mur=mur, amp=amp, phase=phase, 
                        cnorm=cnorm, m=self.mn[0], n=self.mn[1])
            self.add_integrator(engine, 'inc_amp', coeff,
                                b.AddBoundaryIntegrator,
                                mfem.BoundaryLFIntegrator)

        else:
            pass

    def has_extra_DoF(self, kfes):
        if self.fixed_total_field:
            return False
        if kfes == 0 or kfes == 1:
            return True
        return False

    def get_extra_NDoF(self):
        if self.fixed_total_field:
            return 0
        return 1
    
    def is_extra_RHSonly(self):
        return True
    
    def extra_DoF_name2(self, kfes=0):
        '''
        default DoF name
        '''
        if kfes == 0:
            return self.get_root_phys().dep_vars[0]+"_port"+str(self.port_idx)            
        elif kfes == 1:
            return self.get_root_phys().dep_vars[1]+"_port"+str(self.port_idx)

        assert False, "should not come here"

    def get_probes(self):
        if self.fixed_total_field:
            return []
        return [self.get_root_phys().dep_vars[0]+"_port"+str(self.port_idx), 
                self.get_root_phys().dep_vars[1]+"_port"+str(self.port_idx), ]

    def postprocess_extra(self, sol, flag, sol_extra):
        if self.fixed_total_field:
            return
        name = self.name() + '_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()

    def add_extra_contribution(self, engine, **kwargs):
        from mfem.common.chypre import (LF2PyVec, 
                                        PyVec2PyMat, 
                                        Array2PyVec, 
                                        IdentityPyMat)

        kfes = kwargs.pop('kfes', 0)
        fes = engine.get_fes(self.get_root_phys(), kfes)

        dprint1("Add Extra contribution" + str(self._sel_index))

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        Erz, Ephi = self.get_e_coeff_cls()
        Hrz, Hphi = self.get_h_coeff_cls()
        cnorm = self.get_root_phys().get_coeff_norm()

        if kfes == 0:
            lf1 = engine.new_lf(fes)
            Ht = Hrz(2, self, real=True, eps=eps, mur=mur, cnorm=cnorm,
                     m=self.mn[0], n=self.mn[1]) 

            print(Ht)           
            print(isinstance(Ht, mfem.VectorCoefficient))
            Ht = self.restrict_coeff(Ht, engine, vec = True)
            intg =  mfem.VectorFEDomainLFIntegrator(Ht)
            #intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
            lf1.AddBoundaryIntegrator(intg)
            lf1.Assemble()

            lf1i = engine.new_lf(fes)
            Ht = Hrz(2, self, real=False, eps=eps, mur=mur, cnorm=cnorm,
                     m=self.mn[0], n=self.mn[1])                        
            Ht = self.restrict_coeff(Ht, engine, vec = True)
            intg =  mfem.VectorFEDomainLFIntegrator(Ht)
            #intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
            lf1i.AddBoundaryIntegrator(intg)
            lf1i.Assemble()

            lf2r = engine.new_lf(fes)
            Etr = Erz(2, self, real=True, eps=eps, mur=mur,
                      m=self.mn[0], n=self.mn[1])
            Etr = self.restrict_coeff(Etr, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Etr)
            lf2r.AddBoundaryIntegrator(intg)
            lf2r.Assemble()

            lf2i = engine.new_lf(fes)
            Eti = Erz(2, self, real=False, eps=eps, mur=mur,
                      m=self.mn[0], n=self.mn[1])
            Eti = self.restrict_coeff(Eti, engine, vec=True)
            intg = mfem.VectorFEDomainLFIntegrator(Eti)
            lf2i.AddBoundaryIntegrator(intg)
            lf2i.Assemble()

            arr = self.get_restriction_array(engine)

            xr = engine.new_gf(fes)
            xr.Assign(0.0)
            xr.ProjectBdrCoefficientTangent(Etr, arr)
            xi = engine.new_gf(fes)
            xi.Assign(0.0)
            xi.ProjectBdrCoefficientTangent(Eti, arr)

            # transfer x and lf2 to True DoF space to evaluate inner-product
            et = engine.x2X(xr).GetDataArray() + \
                   1j * engine.x2X(xi).GetDataArray()
            vec = engine.b2B(lf2r).GetDataArray() -  \
                   1j * engine.b2B(lf2i).GetDataArray() # complex conjugate
            weight = np.sum(et.dot(vec))
            if use_parallel:
                weight = np.sum(allgather(weight))
                
            t1 = LF2PyVec(lf1, lf1i)
            #v1 *= -1
            t1 = PyVec2PyMat(t1)

            lf2i *= -1 # complex conjugate
            t2 = LF2PyVec(lf2r, lf2i, horizontal=True)
            t2 *= -1. / weight
            t2 = PyVec2PyMat(t2.transpose())
            t2 = t2.transpose()

            t3 = IdentityPyMat(1)

            inc_wave = inc_amp * np.exp(1j * inc_phase / 180. * np.pi)
            phase = np.angle(inc_wave) * 180 / np.pi
            amp = np.sqrt(np.abs(inc_wave))

            t4 = -np.array(
                         [[amp * np.exp(1j * phase / 180. * np.pi)]])
            t4 = Array2PyVec(t4)

            return (t1, t2, t3, t4, True)

        elif kfes == 1:

            lf1 = engine.new_lf(fes)
            Ht = Hphi(self, real=True, eps=eps, mur=mur, cnorm=cnorm,
                      m=self.mn[0], n=self.mn[1])            
            Ht = self.restrict_coeff(Ht, engine)
            intg = mfem.BoundaryLFIntegrator(Ht)
            lf1.AddBoundaryIntegrator(intg)
            lf1.Assemble()

            lf1i = engine.new_lf(fes)
            Ht = Hphi(self, real=False, eps=eps, mur=mur, cnorm=cnorm,
                      m=self.mn[0], n=self.mn[1])
            Ht = self.restrict_coeff(Ht, engine)
            intg = mfem.BoundaryLFIntegrator(Ht)
            lf1i.AddBoundaryIntegrator(intg)
            lf1i.Assemble()

            lf2r = engine.new_lf(fes)
            Etr = Ephi(self, real=True, eps=eps, mur=mur,
                       m=self.mn[0], n=self.mn[1])
            Etr = self.restrict_coeff(Etr, engine)
            intg = mfem.BoundaryLFIntegrator(Etr)
            lf2r.AddBoundaryIntegrator(intg)
            lf2r.Assemble()

            lf2i = engine.new_lf(fes)
            Eti = Ephi(self, real=False, eps=eps, mur=mur,
                       m=self.mn[0], n=self.mn[1])
            Eti = self.restrict_coeff(Eti, engine)
            intg = mfem.BoundaryLFIntegrator(Eti)
            lf2i.AddBoundaryIntegrator(intg)
            lf2i.Assemble()

            arr = self.get_restriction_array(engine)

            xr = engine.new_gf(fes)
            xr.Assign(0.0)
            xr.ProjectBdrCoefficient(Etr, arr)
            xi = engine.new_gf(fes)
            xi.Assign(0.0)
            xi.ProjectBdrCoefficient(Eti, arr)

            # transfer x and lf2 to True DoF space to evaluate inner-product
            et = engine.x2X(xr).GetDataArray() + \
                   1j * engine.x2X(xi).GetDataArray()
            vec = engine.b2B(lf2r).GetDataArray() -  \
                   1j * engine.b2B(lf2i).GetDataArray() # complex conjugate
            weight = np.sum(et.dot(vec))
            if use_parallel:
                weight = np.sum(allgather(weight))

            t1 = LF2PyVec(lf1, lf1i)
            #v1 *= -1
            t1 = PyVec2PyMat(t1)

            lf2i *= -1 # complex conjugate
            t2 = LF2PyVec(lf2r, lf2i, horizontal=True)
            t2 *= -1. / weight
            t2 = PyVec2PyMat(t2.transpose())
            t2 = t2.transpose()

            t3 = IdentityPyMat(1)

            inc_wave = inc_amp * np.exp(1j * inc_phase / 180. * np.pi)
            phase = np.angle(inc_wave) * 180 / np.pi
            amp = np.sqrt(np.abs(inc_wave))

            t4 = -np.array(
                         [[amp * np.exp(1j * phase / 180. * np.pi)]])
            t4 = Array2PyVec(t4)

            return (t1, t2, t3, t4, True)

            '''
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
            '''
