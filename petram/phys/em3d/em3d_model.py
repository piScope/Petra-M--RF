
'''
EM3D : Frequency domain Maxwell equation.

   This module is meant to solve 

    (curl v, curl u) + (v, e u) 
           - (v, n x (1/mu curl u)_s = i w (jext, u) 

    , where e = - epsion * w^2 - i * sigma * w, w is frequency,
    and  mu, epislon, and simga is regular EM parameters.

    ( , ) is volume integral and ( , )_s is surface intengral
    on non-essential boundaries.

    exp(-i w t) is assumed.

 *sigma
  Domain:   
     EM3D_Anisotropic : tensor dielectric
     EM3D_Vac         : scalar dielectric
     EM3D_ExtJ        : external current
     EM3D_Div         : div J = 0 constraints (add Lagrange multiplier)

  Boundary:
     EM3D_PEC         : Perfect electric conductor
     EM3D_PMC         : Perfect magnetic conductor
     EM3D_H           : Mangetic field boundary
     EM3D_SurfJ       : Surface current
     EM3D_Port        : TE, TEM, Coax port
     EM3D_E           : Electric field
     EM3D_Continuity  : Continuitiy

  Pair:
     EM3D_Floquet     : Periodic boundary condition
'''
from petram.phys.vtable import VtableElement, Vtable
import numpy as np
import traceback

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.em3d.em3d_base import EM3D_Bdry
from petram.phys.em3d.em3d_vac import EM3D_Vac


class EM3D_DefDomain(EM3D_Vac):
    can_delete = False
    nlterms = []
    #vt  = Vtable(data1)
    # do not use vtable here, since we want to use
    # vtable defined in EM3D_Vac in add_bf_conttribution

    def __init__(self, **kwargs):
        super(EM3D_DefDomain, self).__init__(**kwargs)

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


class EM3D_DefBdry(EM3D_Bdry):
    can_delete = False
    is_essential = False
    nlterms = []
    vt = Vtable(data2)

    def __init__(self, **kwargs):
        super(EM3D_DefBdry, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_DefBdry, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = ['remaining']
        return v
    '''   
    def panel1_param(self):
        return [['Default Bdry (PMC)',   "Ht = 0",  2, {}],]

    def get_panel1_value(self):
        return None

    def import_panel1_value(self, v):
        pass
    
    def panel1_tip(self):
        return None
    '''

    def get_possible_bdry(self):
        return []


class EM3D_DefPair(Pair, Phys):
    can_delete = False
    is_essential = False

    def __init__(self, **kwargs):
        super(EM3D_DefPair, self).__init__(**kwargs)
        Phys.__init__(self)

    def attribute_set(self, v):
        super(EM3D_DefPair, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        return v

    def get_possible_pair(self):
        return []


class EM3D(PhysModule):
    der_var_base = ['Bx', 'By', 'Bz']
    der_var_vec = ['B']
    geom_dim = 3

    def __init__(self, **kwargs):
        super(EM3D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = EM3D_DefDomain()
        self['Boundary'] = EM3D_DefBdry()
        self['Pair'] = EM3D_DefPair()

    @property
    def dep_vars(self):
        '''
        list of dependent variables, for example.
           [E]      
           [E, psi]
        '''
        ret = ['E']
        if self._has_div_constraint():
            ret = ['E', 'psi']
        else:
            ret = ['E']
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars0(self):
        '''
        list of dependent variables, for example.
           [E]      
           [E, psi]
        '''
        ret = ['E']
        if self._has_div_constraint():
            ret = ['E', 'psi']
        else:
            ret = ['E']
        return [x + self.dep_vars_suffix for x in ret]

    @property
    def dep_vars_base(self):
        ret = ['E']
        if self._has_div_constraint():
            ret = ['E', 'psi']
        else:
            ret = ['E']
        return ret

    def get_fec_type(self, idx):
        values = ['ND', 'H1']
        return values[idx]

    def get_fec(self):
        v = self.dep_vars
        if len(v) == 1:  # normal case
            return [(v[0], 'ND_FECollection')]
        else:            # with divergence constraints
            return [(v[0], 'ND_FECollection'),
                    (v[1], 'H1_FECollection'), ]

    def _has_div_constraint(self):
        from petram.phys.em3d.em3d_div import EM3D_Div
        for mm in self['Domain'].iter_enabled():
            if isinstance(mm, EM3D_Div):
                return True
        return False

    def attribute_set(self, v):
        v = super(EM3D, self).attribute_set(v)
        v["element"] = 'ND_FECollection'
        v["freq_txt"] = "1.0e9"
        v["ndim"] = 3
        v["ind_vars"] = 'x, y, z'
        v["dep_vars_suffix"] = ''
        return v

    def panel1_param(self):
        panels = super(EM3D, self).panel1_param()
        a, b = self.get_var_suffix_var_name_panel()
        panels.extend([self.make_param_panel('freq',  self.freq_txt),
                       ["independent vars.", self.ind_vars, 0, {}],
                       a,
                       ["dep. vars.", ','.join(self.dep_vars), 2, {}],
                       ["ns vars.", ','.join(EM3D.der_var_base), 2, {}], ])

        return panels

    def get_panel1_value(self):
        names = ', '.join(self.dep_vars)
        names2 = ', '.join(list(self.get_default_ns()))
        val = super(EM3D, self).get_panel1_value()
        val.extend([self.freq_txt,
                    self.ind_vars, self.dep_vars_suffix,
                    names, names2, ])
        return val

    """                  
    def attribute_expr(self):
        return ["freq"], [float]
    def attribute_mirror_ns(self):
        return ['freq']
    """

    def get_default_ns(self):
        from petram.phys.phys_const import mu0, epsilon0, q0, massu
        ns = {'mu0': mu0,
              'e0': epsilon0,
              'q0': q0,
              'massu': massu}
        return ns

    def import_panel1_value(self, v):
        v = super(EM3D, self).import_panel1_value(v)
        self.freq_txt = str(v[0])
        self.ind_vars = str(v[1])
        self.dep_vars_suffix = str(v[2])
        #from petram.phys.em3d.em3d_const import mu0, epsilon0
        #self._global_ns['mu0'] = mu0
        #self._global_ns['epsilon0'] = epsilon0

    def get_possible_bdry(self):
        if EM3D._possible_constraints is None:
            self._set_possible_constraints('em3d')
        bdrs = super(EM3D, self).get_possible_bdry()
        return EM3D._possible_constraints['bdry'] + bdrs

    def get_possible_domain(self):
        if EM3D._possible_constraints is None:
            self._set_possible_constraints('em3d')

        doms = super(EM3D, self).get_possible_domain()
        return EM3D._possible_constraints['domain'] + doms

    '''
    def get_possible_bdry(self):
        from .em3d_pec         import EM3D_PEC
        from .em3d_pmc         import EM3D_PMC
        from .em3d_h           import EM3D_H
        from .em3d_surfj       import EM3D_SurfJ
        from .em3d_port        import EM3D_Port
        from .em3d_portarray        import EM3D_PortArray
        from .em3d_e           import EM3D_E
        from .em3d_cont        import EM3D_Continuity
        from .em3d_z           import EM3D_Impedance
        bdrs = super(EM3D, self).get_possible_bdry()
        return [EM3D_PEC, EM3D_Port, EM3D_PortArray, EM3D_E, EM3D_SurfJ, 
                EM3D_H, EM3D_PMC, EM3D_Impedance, EM3D_Continuity]+bdrs
    
    def get_possible_domain(self):
        from .em3d_anisotropic import EM3D_Anisotropic
        from .em3d_vac         import EM3D_Vac
        from .em3d_extj        import EM3D_ExtJ
        from .em3d_div         import EM3D_Div
        
        doms = super(EM3D, self).get_possible_domain()
        
        return [EM3D_Vac, EM3D_Anisotropic, EM3D_ExtJ, EM3D_Div] + doms
    '''

    def get_possible_edge(self):
        return []

    '''
    def get_possible_pair(self):

        from .em3d_floquet     import EM3D_Floquet

        return [EM3D_Floquet]
    '''

    def get_possible_pair(self):
        if EM3D._possible_constraints is None:
            self._set_possible_constraints()
        return EM3D._possible_constraints['pair']

    '''
    def get_possible_point(self):
        return []
    '''

    def is_complex(self):
        return True

    def get_freq_omega(self):
        freq, _void = self.eval_param_expr(self.freq_txt, "freq")
        try:
            _void = float(freq)
        except:
            freq = 1e6
            dprint1("Error, frequency must be a scalr real value")
        return freq, 2.*np.pi*freq

    def add_variables(self, v, name, solr, soli=None):
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_component_expression as addc_expression
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant

        from petram.helper.eval_deriv import eval_curl

        freq, omega = self.get_freq_omega()

        def evalB(gfr, gfi=None):
            gfr, gfi, extra = eval_curl(gfr, gfi)
            gfi /= omega   # real B
            gfr /= -omega  # imag B
            # flipping gfi and gfr so that it returns
            # -i * (-gfr + i gfi) = gfi + i gfr
            return gfi, gfr, extra

        ind_vars = [x.strip()
                    for x in self.ind_vars.split(',') if x.strip() != '']

        suffix = self.dep_vars_suffix

        from petram.helper.variables import TestVariable
        #v['debug_test'] =  TestVariable()

        add_coordinates(v, ind_vars)
        add_surf_normals(v, ind_vars)

        from petram.phys.phys_const import mu0, epsilon0, q0

        if name.startswith('E'):
            freq, omega = self.get_freq_omega()
            add_constant(v, 'freq', suffix, freq)
            add_constant(v, 'omega', suffix, np.float64(omega))
            #add_constant(v, 'mu0', '', mu0)
            #add_constant(v, 'e0', '', epsilon0)

            add_components(v, 'E', suffix, ind_vars, solr, soli)
            add_components(v, 'B', suffix, ind_vars, solr, soli,
                           deriv=evalB)
            add_expression(v, 'normE', suffix, ind_vars,
                           '(conj(Ex)*Ex + conj(Ey)*Ey +conj(Ez)*Ez)**(0.5)',
                           ['E'])

            add_expression(v, 'normB', suffix, ind_vars,
                           '(conj(Bx)*Bx + conj(By)*By + conj(Bz)*Bz)**(0.5)',
                           ['B'])

            # Poynting Flux
            addc_expression(v, 'Poy', suffix, ind_vars,
                            '(conj(Ey)*Bz - conj(Ez)*By)/mu0',
                            ['B', 'E'], 0)
            addc_expression(v, 'Poy', suffix, ind_vars,
                            '(conj(Ez)*Bx - conj(Ex)*Bz)/mu0',
                            ['B', 'E'], 1)
            addc_expression(v, 'Poy', suffix, ind_vars,
                            '(conj(Ex)*By - conj(Ey)*Bx)/mu0',
                            ['B', 'E'], 2)

            #e = - epsion * w^2 - i * sigma * w
            # Jd : displacement current  = -i omega* e0 er E
            addc_expression(v, 'Jd', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[0]',
                            ['epsilonr', 'E', 'freq'],  0)
            addc_expression(v, 'Jd', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[1]',
                            ['epsilonr', 'E', 'freq'], 1)
            addc_expression(v, 'Jd', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[2]',
                            ['epsilonr', 'E', 'freq'], 2)
            # Ji : induced current = sigma *E
            addc_expression(v, 'Ji', suffix, ind_vars,
                            '(dot(sigma, E))[0]',
                            ['sigma', 'E'], 0)
            addc_expression(v, 'Ji', suffix, ind_vars,
                            '(dot(sigma, E))[1]',
                            ['sigma', 'E'], 1)
            addc_expression(v, 'Ji', suffix, ind_vars,
                            '(dot(sigma, E))[2]',
                            ['sigma', 'E'], 2)
            # Jp : polarization current (Jp = -i omega* e0 (er - 1) E
            addc_expression(v, 'Jp', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[0]',
                            ['epsilonr', 'E', 'freq'], 0)
            addc_expression(v, 'Jp', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[1]',
                            ['epsilonr', 'E', 'freq'], 1)
            addc_expression(v, 'Jp', suffix, ind_vars,
                            '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[2]',
                            ['epsilonr', 'E', 'freq'], 2)

            # Js : surface current (n x B / mu)
            addc_expression(v, 'Js', suffix, ind_vars,
                            '(cross([nx, ny, nz], inv(mur).dot(B))/mu0)[0]',
                            ['nx', 'ny', 'nz', 'B', 'mur', 'mu0'], 0)
            addc_expression(v, 'Js', suffix, ind_vars,
                            '(cross([nx, ny, nz], inv(mur).dot(B))/mu0)[1]',
                            ['nx', 'ny', 'nz', 'B', 'mur', 'mu0'], 1)
            addc_expression(v, 'Js', suffix, ind_vars,
                            '(cross([nx, ny, nz], inv(mur).dot(B))/mu0)[2]',
                            ['nx', 'ny', 'nz', 'B', 'mur', 'mu0'], 2)

        elif name.startswith('psi'):
            add_scalar(v, 'psi', suffix, ind_vars, solr, soli)

        # collect all definition from children
        # for mm in self.walk():
        #    if not mm.enabled: continue
        #    if mm is self: continue
        #    mm.add_domain_variables(v, name, suffix, ind_vars,
        #                            solr, soli)
        #    mm.add_bdr_variables(v, name, suffix, ind_vars,
        #                            solr, soli)

        return v

    def get_fes_for_dep(self, unknown_name, soldict):
        keys = soldict.keys()
        for k in keys:
            if unknown_name.startswith('psi'):
                if k.startswith('psi'):
                    break
            else:
                if k.startswith('E'):
                    break
        sol = soldict[k]
        solr = sol[0]
        soli = sol[1] if len(sol) > 1 else None
        return solr, soli
