
'''
EM2D : 2D Frequency domain Maxwell equation.

    This module is meant to solve 

    (curl v, curl u) + (v, e u) 
           - (v, n x (1/mu curl u)_s = i w (jext, u) 

    We decompose the x-y-z filed coponents in in-plane and out-of-plane 
    components, Exy and Ez (respectively).  

    Exy = (Ex, Ey) and  where Exy is Nedelec and Ez is H1

    We use
        curl E = curl Exy + ez x (ik Exy - grad Ez), 
    where k is out-of-plane wave number

    curl-curl is decomposed to 
       (curl Exy, curl Wxy) + (ik Exy - grad Ez,  ik Wxy - grad Wz),

    mass 
       e ((Exy, Wxy) + (Ez, Wz)), 

    where e = - epsion * w^2 - i * sigma * w, w is frequency,
    and  mu, epislon, and simga is regular EM parameters.

    ( , ) is volume integral and ( , )_s is surface intengral
    on non-essential boundaries.

    exp(-i w t) is assumed.

 *sigma
  Domain:   
     EM2D_Anisotropic : tensor dielectric
     EM2D_Vac         : scalar dielectric
     EM2D_ExtJ        : external current

  Boundary:
     EM2D_PEC         : Perfect electric conductor
     EM2D_PMC         : Perfect magnetic conductor
     EM2D_H           : Mangetic field boundary (N.I)
     EM2D_SurfJ       : Surface current         (N.I)
     EM2D_Port        : TE, TEM, Coax port      (N.I)
     EM2D_E           : Electric field
     EM2D_Continuity  : Continuitiy
     EM2D_Z           : Impedance

'''
import numpy as np
import traceback

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.em2d.em2d_base import EM2D_Bdry
from petram.phys.em2d.em2d_vac import EM2D_Vac


from petram.phys.vtable import VtableElement, Vtable


class EM2D_DefDomain(EM2D_Vac):
    can_delete = False
    nlterms = []
    #vt  = Vtable(data1)
    # do not use vtable here, since we want to use
    # vtable defined in EM2D_Vac in add_bf_conttribution

    def __init__(self, **kwargs):
        super(EM2D_DefDomain, self).__init__(**kwargs)

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

    def get_possible_child(self):
        return self.parent.get_possible_domain()


data2 = (('label1', VtableElement(None,
                                  guilabel='Default Bdry (PMC)',
                                  default="Ht = 0",
                                  tip="this is a natural BC")),)


class EM2D_DefBdry(EM2D_Bdry):
    can_delete = False
    is_essential = False
    nlterms = []
    vt = Vtable(data2)

    def __init__(self, **kwargs):
        super(EM2D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM2D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v

    def get_possible_bdry(self):
        return []


class EM2D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False
    is_complex = True

    def __init__(self, **kwargs):
        super(EM2D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM2D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_possible_pair(self):
        return []


class EM2D(PhysModule):
    der_vars_base = ['Bx', 'By', 'Bz']
    der_vars_vec = ['E', 'B']
    geom_dim = 2

    def __init__(self, **kwargs):
        super(EM2D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = EM2D_DefDomain()
        self['Boundary'] = EM2D_DefBdry()
        self['Pair'] = EM2D_DefPair()

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
           [Exy, Ez]      
           [Exy, Ez, psi]
        '''
        ret = self.dep_vars_base
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars_base(self):
        if self._has_div_constraint():
            ret = ['Exy', 'Ez', 'psi']
        else:
            ret = ['Exy', 'Ez']
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

    def attribute_set(self, v):
        v = super(EM2D, self).attribute_set(v)
        v["element"] = 'ND_FECollection, H1_FECollection'
        v["freq_txt"] = 1.0e9
        v["ndim"] = 2
        v["ind_vars"] = 'x, y'
        v["dep_vars_suffix"] = ''
        return v

    def panel1_param(self):
        panels = super(EM2D, self).panel1_param()
        panels.extend([self.make_param_panel('freq',  self.freq_txt),
                       ["independent vars.", self.ind_vars, 0, {}],
                       ["dep. vars. suffix", self.dep_vars_suffix, 0, {}],
                       ["dep. vars.", ','.join(self.dep_vars), 2, {}],
                       ["derived vars.", ','.join(EM2D.der_vars_base), 2, {}], ])

        return panels

    def get_panel1_value(self):
        names = ', '.join([x for x in self.dep_vars])
        names2 = ', '.join(self.get_dependent_variables())
        val = super(EM2D, self).get_panel1_value()
        val.extend([self.freq_txt, self.ind_vars, self.dep_vars_suffix,
                    names, names2, ])
        return val

    """
    def attribute_expr(self):
        return ["freq"], [float]
    def attribute_mirror_ns(self):
        return ['freq']
    """

    def get_default_ns(self):
        from petram.phys.phys_const import mu0, epsilon0, q0
        ns = {'mu0': mu0,
              'e0': epsilon0,
              'q0': q0}
        return ns

    def import_panel1_value(self, v):
        v = super(EM2D, self).import_panel1_value(v)
        self.freq_txt = str(v[0])
        self.ind_vars = str(v[1])
        self.dep_vars_suffix = str(v[2])

        from petram.phys.phys_const import mu0, epsilon0, q0
        self._global_ns['mu0'] = mu0
        self._global_ns['epsilon0'] = epsilon0

    def get_possible_bdry(self):
        if EM2D._possible_constraints is None:
            self._set_possible_constraints('em2d')
        bdrs = super(EM2D, self).get_possible_bdry()
        return EM2D._possible_constraints['bdry'] + bdrs

    def get_possible_domain(self):
        if EM2D._possible_constraints is None:
            self._set_possible_constraints('em2d')

        doms = super(EM2D, self).get_possible_domain()
        return EM2D._possible_constraints['domain'] + doms

    def get_possible_edge(self):
        return []

    def get_possible_pair(self):
        if EM2D._possible_constraints is None:
            self._set_possible_constraints('em2d')

        pairs = super(EM2D, self).get_possible_pair()
        return EM2D._possible_constraints['pair'] + pairs

    def get_possible_point(self):
        return []

    def is_complex(self):
        return True

    def get_freq_omega(self):
        return self._global_ns['freq'], 2.*np.pi*self._global_ns['freq']

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

        def eval_curlExy(gfr, gfi=None):
            gfr, gfi, extra = eval_curl(gfr, gfi)
            return gfr, gfi, extra

        def eval_gradEz(gfr, gfi=None):
            gfr, gfi, extra = eval_grad(gfr, gfi)
            return gfr, gfi, extra

        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        suffix = self.dep_vars_suffix

        freq, omega = self.get_freq_omega()
        add_constant(v, 'omega', suffix, np.float64(omega),)
        add_constant(v, 'freq', suffix, np.float64(freq),)
        #add_constant(v, 'mu0', '', self._global_ns['mu0'])
        #add_constant(v, 'e0', '', self._global_ns['e0'])

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)

        if name.startswith('Exy'):
            add_elements(v, 'E', suffix, ind_vars, solr, soli, elements=[0, 1])
            add_scalar(v, 'curlExy', suffix, ind_vars, solr, soli,
                       deriv=eval_curlExy)
            addc_expression(v, 'B', suffix, ind_vars,
                            '-1j/omega*curlExy', ['curlExy', 'omega'], 'z')

        elif name.startswith('Ez'):
            add_scalar(v, 'Ez', suffix, ind_vars, solr, soli, vars=['E'])
            add_components(v, 'gradE', suffix, ind_vars, solr, soli,
                           deriv=eval_gradEz)

        elif name.startswith('psi'):
            add_scalar(v, 'psi', suffix, ind_vars, solr, soli)

        add_expression(v, 'E', suffix, ind_vars, 'array([Ex, Ey, Ez])',
                       ['Ex', 'Ey', 'Ez'])

        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(-1j*kz*Ey + gradEy)',
                        ['m_mode', 'E', 'omega'], 0)
        addc_expression(v, 'B', suffix, ind_vars,
                        '-1j/omega*(1j*kz*Ex - gradEx)',
                        ['m_mode', 'E', 'omega'], 1)
        add_expression(v, 'B', suffix, ind_vars,
                       'array([Bx, By, Bz])',
                       ['B'])

        # Poynting Flux

        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ey)*Bz - conj(Ez)*By)/mu0',
                        ['B', 'E'], 0)
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ez)*Bx - conj(Ex)*Bz)/mu0',
                        ['B', 'E'], 'y')
        addc_expression(v, 'Poy', suffix, ind_vars,
                        '(conj(Ex)*By - conj(Ey)*Bx)/mu0',
                        ['B', 'E'], 'z')

        return v

    def get_fes_for_dep(self, unknown_name, soldict):
        keys = soldict.keys()
        for k in keys:
            if unknown_name.startswith('phi'):
                if k.startswith('psi'):
                    break
            elif unknown_name.startswith('Et'):
                if k.startswith('Exy'):
                    break
            else:
                if k.startswith('Ez'):
                    break
        sol = soldict[k]
        solr = sol[0]
        soli = sol[1] if len(sol) > 1 else None
        return solr, soli
