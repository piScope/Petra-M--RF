
'''
EM2Da : Axis- symmetric Frequency domain Maxwell equation.

    This module is meant to solve 

    (curl v, curl u) + (v, e u) 
           - (v, n x (1/mu curl u)_s = i w (jext, u) 

    We decompose the r-phi-z filed coponents in toroidal and poloidal 
    components, Et and Ep. Note that Ep is a vector and Et is not Ephi

    Ep = (Er, Ez) and Et= rEphi, where Ep is Nedelec and Et is H1

    curl-curl is decomposed to 
       r (curl Ep, curl Wp) + 1/r (m Et - div Ep,  m Wt - div Wp),

    mass 
       e (r (Ep, Wp) + 1/r (Et, Wt)), 

    where e = - epsion * w^2 - i * sigma * w, w is frequency,
    and  mu, epislon, and simga is regular EM parameters.

    ( , ) is volume integral and ( , )_s is surface intengral
    on non-essential boundaries.

    exp(-i w t) is assumed.

 *sigma
  Domain:   
     EM2Da_Anisotropic : tensor dielectric
     EM2Da_Vac         : scalar dielectric
     EM2Da_ExtJ        : external current
     EM2Da_Div         : div J = 0 constraints (add Lagrange multiplier)

  Boundary:
     EM2Da_PEC         : Perfect electric conductor
     EM2Da_PMC         : Perfect magnetic conductor
     EM2Da_H           : Mangetic field boundary
     EM2Da_SurfJ       : Surface current
     EM2Da_Port        : TE, TEM, Coax port
     EM2Da_E           : Electric field
     EM2Da_Continuity  : Continuitiy

'''
import numpy as np
import traceback

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys
from petram.phys.common.em_base import EMPhysModule
from petram.phys.em2da.em2da_base import EM2Da_Bdry
from petram.phys.em2da.em2da_vac import EM2Da_Vac

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2DaModel')

from petram.phys.vtable import VtableElement, Vtable

class EM2Da_DefDomain(EM2Da_Vac):
    can_delete = False
    nlterms = []
    #vt  = Vtable(data1)
    # do not use vtable here, since we want to use
    # vtable defined in EM3D_Vac in add_bf_conttribution

    def __init__(self, **kwargs):
        super(EM2Da_DefDomain, self).__init__(**kwargs)

    def panel1_param(self):
        return [['Default Domain (Vac)',   "eps_r=1, mu_r=1, sigma=0",  2, {}], ]

    def get_panel1_value(self):
        return ["eps_r=1, mu_r=1, sigma=0", ]

    def import_panel1_value(self, v):
        pass

    def panel1_tip(self):
        return None

    def get_possible_domain(self):
        return []


data2 = (('label1', VtableElement(None,
                                  guilabel='Default Bdry (PMC)',
                                  default="Ht = 0",
                                  tip="this is a natural BC")),)


class EM2Da_DefBdry(EM2Da_Bdry):
    can_delete = False
    is_essential = False
    nlterms = []
    vt = Vtable(data2)

    def __init__(self, **kwargs):
        super(EM2Da_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM2Da_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class EM2Da_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = True

    def __init__(self, **kwargs):
        super(EM2Da_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM2Da_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_possible_pair(self):
        return []


class EM2Da(EMPhysModule):
    der_vars_base = ['Er', 'Ephi', 'Ez', 'Br', 'Bphi', 'Bz', 'm_mode']
    der_vars_vec = ['E', 'B']
    geom_dim = 2

    def __init__(self, **kwargs):
        super(EM2Da, self).__init__(**kwargs)
        self['Domain'] = EM2Da_DefDomain()
        self['Boundary'] = EM2Da_DefBdry()
        self['Pair'] = EM2Da_DefPair()

    @property
    def dep_vars(self):
        '''
        list of dependent variables, for example.
           [Et, rEf]      
           [Et, rEf, psi]
        '''
        ret = self.dep_vars_base
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars0(self):
        '''
        list of dependent variables, for example.
           [Et, rEf]      
           [Et, rEf, psi]
        '''
        ret = self.dep_vars_base
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars_base(self):
        if self._has_div_constraint():
            ret = ['Et', 'rEf', 'psi']
        else:
            ret = ['Et', 'rEf']
        return ret

    def get_fec_type(self, idx):
        values = ['ND', 'H1', 'H1']
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        if len(v) == 2:  # normal case
            return [(v[0], 'ND_FECollection'),
                    (v[1], 'H1_FECollection'), ]
        else:            # with divergence constraints
            return [(v[0], 'ND_FECollection'),
                    (v[1], 'H1_FECollection'),
                    (v[2], 'H1_FECollection'), ]

    def _has_div_constraint(self):
        return False
        #from .em2da_div import EM2Da_Div
        # for mm in self['Domain'].iter_enabled():
        #    if isinstance(mm, EM2Da_Div): return True
        # return False

    def attribute_set(self, v):
        v = super(EM2Da, self).attribute_set(v)
        v["element"] = 'ND_FECollection, H1_FECollection'
        v["ndim"] = 2
        v["ind_vars"] = 'r, z'
        v["dep_vars_suffix"] = ''
        return v

    def panel1_param(self):
        panels = super(EM2Da, self).panel1_param()
        panels.extend(#[self.make_param_panel('freq',  self.freq_txt),
                       ["independent vars.", self.ind_vars, 0, {}],
                       ["dep. vars. suffix", self.dep_vars_suffix, 0, {}],
                       ["dep. vars.", ','.join(self.dep_vars), 2, {}],
                       ["ns vars.", ','.join(EM2Da.der_vars_base), 2, {}], ])

        return panels

    def get_panel1_value(self):
        names = ', '.join([x for x in self.dep_vars])
        names2 = ', '.join(list(self.get_default_ns()))
        val = super(EM2Da, self).get_panel1_value()
        #val.extend([self.freq_txt, self.ind_vars, self.dep_vars_suffix,
        #            names, names2, ])
        val.extend([self.ind_vars, self.dep_vars_suffix,
                    names, names2, ])
        
        return val

    def import_panel1_value(self, v):
        v = super(EM2Da, self).import_panel1_value(v)
        #self.freq_txt = str(v[0])
        self.ind_vars = str(v[0])
        self.dep_vars_suffix = str(v[1])

    def get_possible_domain(self):
        if EM2Da._possible_constraints is None:
            self._set_possible_constraints('em2da')

        doms = super(EM2Da, self).get_possible_domain()
        return EM2Da._possible_constraints['domain'] + doms

    def get_possible_bdry(self):
        if EM2Da._possible_constraints is None:
            self._set_possible_constraints('em2da')
        bdrs = super(EM2Da, self).get_possible_bdry()
        return EM2Da._possible_constraints['bdry'] + bdrs

    def get_possible_edge(self):
        return []

    def get_possible_pair(self):
        if EM2Da._possible_constraints is None:
            self._set_possible_constraints('em2da')

        pairs = super(EM2Da, self).get_possible_pair()
        return EM2Da._possible_constraints['pair'] + pairs

    def get_possible_point(self):
        return []

    def add_variables(self, v, name, solr, soli=None):
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_elements
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_component_expression as addc_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant

        from petram.phys.em2da.eval_deriv import eval_curl, eval_grad

        v = super(EM2Da, self).add_variables(v, name, solr, soli)

        def eval_curlEt(gfr, gfi=None):
            gfr, gfi, extra = eval_curl(gfr, gfi)
            return gfr, gfi, extra

        def eval_gradrEf(gfr, gfi=None):
            gfr, gfi, extra = eval_grad(gfr, gfi)
            return gfr, gfi, extra

        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        suffix = self.dep_vars_suffix

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)

        if name.startswith('Et'):
            add_elements(v, 'E', suffix, ind_vars, solr, soli, elements=[0, 1])
            add_scalar(v, 'curlEt', suffix, ind_vars, solr, soli,
                       deriv=eval_curlEt)
            addc_expression(v, 'B', suffix, ind_vars,
                            '-1j/omega*(-1)*curlEt', ['curlEt', 'omega'], 'phi')

        elif name.startswith('rEf'):
            add_scalar(v, 'rEf', suffix, ind_vars, solr, soli)
            addc_expression(v, 'E', suffix, ind_vars,
                            'rEf/r', ['rEf', ], 'phi')
            add_components(v, 'gradrE', suffix, ind_vars, solr, soli,
                           deriv=eval_gradrEf)
        elif name.startswith('psi'):
            add_scalar(v, 'psi', suffix, ind_vars, solr, soli)

        add_expression(v, 'E', suffix, ind_vars,
                       'array([Er, Ephi, Ez])',
                       ['E'])

        addc_expression(v, 'E', suffix, ind_vars,
                        'rEf/r', ['rEf', ], 'phi')
        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(1j*m_mode*Ez/r-gradrEz/r)',
                        ['m_mode', 'E', 'omega'], 0)
        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(-1j*m_mode*Er/r+gradrEr/r)',
                        ['m_mode', 'E', 'omega'], 1)
        add_expression(v, 'B', suffix, ind_vars,
                       'array([Br, Bphi, Bz])',
                       ['B'])

        # Poynting Flux
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ephi)*Bz - conj(Ez)*Bphi)/mu0',
                        ['B', 'E'], 0)
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ez)*Br - conj(Er)*Bz)/mu0',
                        ['B', 'E'], 'phi')
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Er)*Bphi - conj(Ephi)*Br)/mu0',
                        ['B', 'E'], 1)

        # collect all definition from children
        '''
        for mm in self.walk():
            if not mm.enabled: continue
            if mm is self: continue
            mm.add_domain_variables(v, name, suffix, ind_vars,
                                    solr, soli)
            mm.add_bdr_variables(v, name, suffix, ind_vars,
                                    solr, soli)
        '''
        return v

    def get_fes_for_dep(self, unknown_name, soldict):
        keys = soldict.keys()
        for k in keys:
            if unknown_name.startswith('phi'):
                if k.startswith('psi'):
                    break
            elif unknown_name.startswith('Et'):
                if k.startswith('Et'):
                    break
            else:
                if k.startswith('rEf'):
                    break
        sol = soldict[k]
        solr = sol[0]
        soli = sol[1] if len(sol) > 1 else None
        return solr, soli
