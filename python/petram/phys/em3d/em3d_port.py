from petram.phys.vtable import VtableElement, Vtable
from petram.phys.common.rf_port_geometry import (analyze_geom_norm_ref,
                                                 analyze_rect_geom,
                                                 analyze_coax_geom,
                                                 analyze_circular_geom)

from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain
from petram.phys.phys_model import Phys
from petram.model import Bdry
from petram.mfem_config import use_parallel
'''

   3D port boundary condition


    2016 5/20  first version only TE modes
'''

import numpy as np

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Port')


if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    from petram.helper.mpi_recipes import *
else:
    import mfem.ser as mfem
    nicePrint = dprint1


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


def bdry_constraints():
    return [EM3D_Port]


class EM3D_Port(EM3D_Bdry):
    extra_diagnostic_print = True
    vt = Vtable(data)

    def __init__(self, mode='TE', mn='0,1', inc_amp='1',
                 inc_phase='0', port_idx=1):
        super(EM3D_Port, self).__init__(mode=mode,
                                        mn=mn,
                                        inc_amp=inc_amp,
                                        inc_phase=inc_phase,
                                        port_idx=port_idx)

    def extra_DoF_name(self):
        return self.get_root_phys().dep_vars[0] + "_port_" + str(self.port_idx)

    def get_probes(self):
        if self.fixed_total_field:
            return []

        return [self.get_root_phys().dep_vars[0] + "_port_" + str(self.port_idx)]

    def attribute_set(self, v):
        super(EM3D_Port, self).attribute_set(v)
        v['port_idx'] = 1
        v['mode'] = 'TE'
        v['mn'] = [1, 0]
        v['inc_amp_txt'] = '1.0'
        v['inc_phase_txt'] = '0.0'
        v['inc_amp'] = 1.0
        v['inc_phase'] = 0.0
        v['epsilonr'] = 1.0
        v['mur'] = 1.0
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['isTimeDependent_RHS'] = True
        v['ref_pt'] = '1'
        v['fixed_total_field'] = False
        return v

    def panel1_param(self):
        from petram.phys.common.rf_portmode import port_modes
        return ([["port id", str(self.port_idx), 0, {}],
                 ["mode", self.mode, 4, {"readonly": True,
                                         "choices": port_modes}],
                 ["m/n", ','.join(str(x) for x in self.mn), 0, {}],
                 ["ref. pt.", "", 0, {}],
                 ["fixed excitaiton", True, 3, {"text": ""}],] +
                self.vt.panel_param(self))

    def get_panel1_value(self):
        return ([str(self.port_idx),
                 self.mode, ','.join(str(x) for x in self.mn),
                 self.ref_pt,
                 self.fixed_total_field] +
                self.vt.get_panel_value(self))

    def import_panel1_value(self, v):
        self.port_idx = v[0]
        self.mode = v[1]
        self.mn = [int(x) for x in v[2].split(',')]
        self.ref_pt = v[3]
        self.fixed_total_field = v[4]
        self.vt.import_panel_value(self, v[5:])

    def panel4_param(self):
        ll = super(EM3D_Port, self).panel4_param()
        ll.append(['Varying (in time/for loop) RHS', False, 3, {"text": ""}])
        return ll

    def panel4_tip(self):
        return None

    def import_panel4_value(self, value):
        super(EM3D_Port, self).import_panel4_value(value[:-1])
        self.isTimeDependent_RHS = value[-1]

    def get_panel4_value(self):
        value = super(EM3D_Port, self).get_panel4_value()
        value.append(self.isTimeDependent_RHS)
        return value

    def verify_setting(self):
        if self.isTimeDependent_RHS:
            flag = True
        else:
            flag = False
        return flag, 'Varying RHS is not set', 'This potntially causes an error with PortScan. Set it Time/NL Dep. panel '

    def update_param(self):
        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        '''
        self.update_inc_amp_phase()

    def update_inc_amp_phase(self):
        try:
            self.inc_amp = self.eval_phys_expr(str(self.inc_amp_txt),  'inc_amp')[0]
            self.inc_phase = self.eval_phys_expr(str(self.inc_phase_txt), 'inc_phase', chk_float = True)[0]
        except:
            raise ValueError("Cannot evaluate amplitude/phase to float number")
        '''

    def preprocess_params(self, engine):
        # find normal (outward) vector...
        mesh = engine.get_emesh(mm=self)

        # fespace = engine.fespaces[self.get_root_phys().dep_vars[0]]

        nbe = mesh.GetNBE()
        ibe = np.array([i for i in range(nbe)
                        if mesh.GetBdrElement(i).GetAttribute() ==
                        self._sel_index[0]])

        norm, rptx = analyze_geom_norm_ref(mesh, ibe, self.ref_pt)
        self.norm = norm

        dprint1("Normal Vector " + str(self.norm))
        dprint1("Ref. Point" + str(rptx))

        if str(self.mode).upper().strip() in ['TE', 'TM', 'TEM']:
            geom_data, vv = analyze_rect_geom(self, mesh, ibe, norm)
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
            geom_data, vv = analyze_coax_geom(mesh, ibe, norm, rptx)
            self.a = geom_data["a"]
            self.b = geom_data["b"]
            self.ctr = geom_data["ctr"]
            self.ax1 = geom_data["ax1"]
            self.ax2 = geom_data["ax2"]

        elif self.mode == 'Circular(TE)':
            geom_data, vv = analyze_circular_geom(mesh, ibe, norm, rptx)
            self.a = geom_data["a"]
            self.ctr = geom_data["ctr"]
            self.ax1 = geom_data["ax1"]
            self.ax2 = geom_data["ax2"]
        else:
            assert False, "unknown mode"

        C_Et, C_jwHt = self.get_coeff_cls()

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)
        dprint1("E field pattern", eps, mur)
        Et = C_Et(3, self, real=True, eps=eps, mur=mur,
                  m=self.mn[0], n=self.mn[1])
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Et.EvalValue(p).__repr__())
        dprint1("H field pattern")
        cnorm = self.get_root_phys().get_coeff_norm()
        Ht = C_jwHt(3, self, real=False, eps=eps, mur=mur, cnorm=cnorm,
                    m=self.mn[0], n=self.mn[1])
        for p in vv:
            dprint1(p.__repr__() + ' : ' + Ht.EvalValue(p).__repr__())

    def get_coeff_cls(self):
        from petram.phys.common.rf_portmode import get_portmode_coeff_cls

        return get_portmode_coeff_cls(self.mode)

    def has_lf_contribution(self, kfes):
        if kfes != 0:
            return False
        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        return inc_amp != 0

    def add_lf_contribution(self, engine, b, real=True, kfes=0):
        if real:
            dprint1("Add LF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add LF contribution(imag)" + str(self._sel_index))

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        dprint1("Power, Phase: ", inc_amp, inc_phase)

        C_Et, C_jwHt = self.get_coeff_cls()

        inc_wave = inc_amp * np.exp(1j * inc_phase / 180. * np.pi)

        phase = np.angle(inc_wave) * 180 / np.pi
        amp = np.sqrt(np.abs(inc_wave))
        cnorm = self.get_root_phys().get_coeff_norm()

        Ht = C_jwHt(3, self, real=real, eps=eps, mur=mur, amp=amp, phase=phase, cnorm=cnorm,
                    m=self.mn[0], n=self.mn[1])
        Ht = self.restrict_coeff(Ht, engine, vec=True)

        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht)
        b.AddBoundaryIntegrator(intg)

    def has_extra_DoF(self, kfes):
        if self.fixed_total_field:
            return False
        if kfes != 0:
            return False
        return True

    def get_extra_NDoF(self):
        if self.fixed_total_field:
            return 0
        return 1

    def is_extra_RHSonly(self):
        return True

    def postprocess_extra(self, sol, flag, sol_extra):
        if self.fixed_total_field:
            return

        name = self.name() + '_' + str(self.port_idx)
        sol_extra[name] = sol.toarray()

    def check_extra_update(self, mode):
        '''
        mode = 'B' or 'M'
        'M' return True, if M needs to be updated
        'B' return True, if B needs to be updated
        '''
        if self._update_flag:
            if mode == 'B':
                return self.isTimeDependent_RHS
            if mode == 'M':
                return self.isTimeDependent
        return False

    def add_extra_contribution(self, engine, **kwargs):
        dprint1("Add Extra contribution" + str(self._sel_index))
        from mfem.common.chypre import LF2PyVec, PyVec2PyMat, Array2PyVec, IdentityPyMat

        self.vt.preprocess_params(self)
        inc_amp, inc_phase, eps, mur = self.vt.make_value_or_expression(self)

        C_Et, C_jwHt = self.get_coeff_cls()
        cnorm = self.get_root_phys().get_coeff_norm()

        fes = engine.get_fes(self.get_root_phys(), 0)

        #
        #  evaulate vectors for t1
        #
        lf1 = engine.new_lf(fes)
        Ht1 = C_jwHt(3, self, real=True, eps=eps, mur=mur, cnorm=cnorm,
                     m=self.mn[0], n=self.mn[1])
        Ht2 = self.restrict_coeff(Ht1, engine, vec=True)
        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht2)
        lf1.AddBoundaryIntegrator(intg)
        lf1.Assemble()
        lf1i = engine.new_lf(fes)
        Ht3 = C_jwHt(3, self, real=False, eps=eps, mur=mur, cnorm=cnorm,
                     m=self.mn[0], n=self.mn[1])
        Ht4 = self.restrict_coeff(Ht3, engine, vec=True)
        intg = mfem.VectorFEBoundaryTangentLFIntegrator(Ht4)
        lf1i.AddBoundaryIntegrator(intg)
        lf1i.Assemble()

        #
        #  evaulate vectors for t2
        #
        lf2r = engine.new_lf(fes)
        Etr = C_Et(3, self, real=True, eps=eps, mur=mur,
                   m=self.mn[0], n=self.mn[1])
        Etr = self.restrict_coeff(Etr, engine, vec=True)
        intg = mfem.VectorFEDomainLFIntegrator(Etr)
        lf2r.AddBoundaryIntegrator(intg)
        lf2r.Assemble()

        lf2i = engine.new_lf(fes)
        Eti = C_Et(3, self, real=False, eps=eps, mur=mur,
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
            1j * engine.b2B(lf2i).GetDataArray()  # complex conjugate
        weight = np.sum(et.dot(vec))
        if use_parallel:
            weight = np.sum(allgather(weight))

        # nicePrint("wegiht ", weight)

        #  t1
        t1 = LF2PyVec(lf1, lf1i)
        # t1 *= -1
        t1 = PyVec2PyMat(t1)

        # t2
        lf2i *= -1  # complex conjugate
        t2 = LF2PyVec(lf2r, lf2i, horizontal=True)
        t2 *= -1. / weight
        t2 = PyVec2PyMat(t2.transpose())
        t2 = t2.transpose()

        # t3
        t3 = IdentityPyMat(1)
        # t3 = IdentityPyMat(1, diag=-1)
        # t4
        inc_wave = inc_amp * np.exp(1j * inc_phase / 180. * np.pi)
        phase = np.angle(inc_wave) * 180 / np.pi
        amp = np.sqrt(np.abs(inc_wave))

        t4 = -np.array(
            [[amp * np.exp(1j * phase / 180. * np.pi)]])
        t4 = Array2PyVec(t4)

        '''
        Format of extar   (t2 is returnd as vertical(transposed) matrix)
        [M,  t1]   [  ]
        [      ] = [  ]
        [t2, t3]   [t4]

        and it returns if Lagurangian will be saved.
        '''
        return (t1, t2, t3, t4, True)

        # return (None, t2, t3, t4, True)
