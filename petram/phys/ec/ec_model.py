
'''
EC3D : Electric Circilt

   This module is meant to solve 

 *sigma
  Domain:   
     EC3D_Conductor   : scalar dielectric
     EC3D_ExtJ        : external current

  Boundary:
     EC3D_Insulation  : Insulation
     EC3D_Potentail   : Potentai
     EC3D_Ground      : Potentai = 0
     EC3D_Continuity  : Continuitiy

'''
import numpy as np
import traceback

from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model import Phys, PhysModule
from petram.phys.em3d.em3d_base import EM3D_Bdry
from petram.phys.em3d.em3d_vac import EM3D_Vac

txt_predefined = 'freq, e0, mu0'

from petram.phys.vtable import VtableElement, Vtable

data2 =  (('label1', VtableElement(None, 
                                     guilabel = 'Default Bdry (PMC)',
                                     default =   "Ht = 0",
                                     tip = "this is a natural BC" )),)
class EM3D_DefDomain(EM3D_Vac):
    can_delete = False
    nlterms = []
    #vt  = Vtable(data1)
    #do not use vtable here, since we want to use
    #vtable defined in EM3D_Vac in add_bf_conttribution
    
    def __init__(self, **kwargs):
        super(EM3D_DefDomain, self).__init__(**kwargs)
    def panel1_param(self):
        return [['Default Domain (Vac)',   "eps_r=1, mu_r=1, sigma=0",  2, {}],]
    def get_panel1_value(self):
        return None
    def import_panel1_value(self, v):
        pass
    def panel1_tip(self):
        return None
    def get_possible_domain(self):
        return []
          
        
class EM3D_DefBdry(EM3D_Bdry):
    can_delete = False
    is_essential = False
    nlterms = []          
    vt  = Vtable(data2)                    
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
    is_complex = True
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

class EC3D(PhysModule):
    der_var_base = ['Jx' 'Jy', 'Jz']
    geom_dim = 3
    def __init__(self, **kwargs):
        super(EC3D, self).__init__()
        Phys.__init__(self)
        self['Domain'] = EC3D_DefDomain()
        self['Boundary'] = EC3D_DefBdry()
        self['Pair'] = EC3D_DefPair()
        
    @property
    def dep_vars(self):
        '''
        list of dependent variables, for example.
           [V]      
        '''
        ret = ['V']
        return [x + self.dep_vars_suffix for x in ret]            
    @property 
    def dep_vars_base(self):
        return  = ['V']

    def get_fec(self):
        v = self.dep_vars
        return [(v[0], 'H1_FECollection')]
    
    def attribute_set(self, v):
        v = super(EC3D, self).attribute_set(v)
        v["element"] = 'H1_FECollection'
        v["freq_txt"]    = 1.0e9
        v["ind_vars"] = 'x, y, z'
        v["dep_vars_suffix"] = ''
        return v
    
    def panel1_param(self):
        panels = super(EC3D, self).panel1_param()
        panels.extend([
                ["independent vars.", self.ind_vars, 0, {}],
                ["dep. vars. suffix", self.dep_vars_suffix, 0, {}],
                ["dep. vars.", ','.join(self.dep_vars), 2, {}],
                ["derived vars.", ','.join(EC3D.der_var_base), 2, {}],
                ["predefined ns vars.", txt_predefined , 2, {}]])
        return panels
      
    def get_panel1_value(self):
        names  = ','.join([x+self.dep_vars_suffix for x in self.dep_vars])
        names2  = ','.join([x+self.dep_vars_suffix for x in EM3D.der_var_base])
        val =  super(EC3D, self).get_panel1_value()
        val.extend([self.ind_vars, self.dep_vars_suffix,
                    names, names2, txt_predefined])
        return val
    
    def get_panel2_value(self):
        return 'all'
    
    def get_default_ns(self):
        from em3d_const import mu0, epsilon0, q0
        ns =  {'mu0': mu0,
               'e0': epsilon0,
               'q0': q0}
        return ns
    
    def import_panel1_value(self, v):
        v = super(EC3D, self).import_panel1_value(v)
        self.freq_txt =  str(v[0])
        self.ind_vars =  str(v[0])
        self.dep_vars_suffix =  str(v[2])                
        from em3d_const import mu0, epsilon0
        self._global_ns['mu0'] = mu0
        self._global_ns['epsilon0'] = epsilon0
            
    def import_panel2_value(self, v):
        self.sel_index = 'all'
      
    def get_possible_bdry(self):
        from em3d_pec       import EM3D_PEC
        from em3d_pmc       import EM3D_PMC
        from em3d_h       import EM3D_H
        from em3d_surfj       import EM3D_SurfJ
        from em3d_port       import EM3D_Port
        from em3d_e       import EM3D_E
        from em3d_cont       import EM3D_Continuity
        return [EM3D_PEC, EM3D_Port, EM3D_E, EM3D_SurfJ, 
                EM3D_H, EM3D_PMC, EM3D_Continuity]
    
    def get_possible_domain(self):
        from ec3d_domains   import EC3D_Conductor, EC3D_ExtJ

        return [EC3D_Conductor, EM3D_ExtJ]

    def get_possible_edge(self):
        return []                

    def get_possible_pair(self):
        return []

    def get_possible_point(self):
        return []

    def is_complex(self):
        return True

    def add_variables(self, v, name, solr, soli = None):
        from petram.helper.variables import add_coordinates
        from petram.helper.variables import add_scalar
        from petram.helper.variables import add_components
        from petram.helper.variables import add_expression
        from petram.helper.variables import add_surf_normals
        from petram.helper.variables import add_constant      
        
        ind_vars = [x.strip() for x in self.ind_vars.split(',')]
        suffix = self.dep_vars_suffix

        from petram.helper.variables import TestVariable
        #v['debug_test'] =  TestVariable()
        
        add_coordinates(v, ind_vars)        
        add_surf_normals(v, ind_vars)
        
        if name.startswith('E'):
            #add_constant(v, 'freq', suffix, self._global_ns['freq'])
            #add_constant(v, 'mu0',  suffix, self._global_ns['mu0'])
            #add_constant(v, 'e0',  suffix, self._global_ns['e0'])
                           
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
            add_expression(v, 'Poyx', suffix, ind_vars,
                           '(conj(Ey)*Bz - conj(Ez)*By)/mu0', ['B', 'E'])
            add_expression(v, 'Poyy', suffix, ind_vars,
                           '(conj(Ez)*Bx - conj(Ex)*Bz)/mu0', ['B', 'E'])
            add_expression(v, 'Poyz', suffix, ind_vars,
                           '(conj(Ex)*By - conj(Ey)*Bx)/mu0', ['B', 'E'])

            #e = - epsion * w^2 - i * sigma * w
            # Jd : displacement current  = -i omega* e0 er E
            add_expression(v, 'Jdx', suffix, ind_vars,
                           '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[0]', 
                           ['epsilonr', 'E'])
            add_expression(v, 'Jdy', suffix, ind_vars,
                           '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[1]', 
                           ['epsilonr', 'E'])
            add_expression(v, 'Jdz', suffix, ind_vars,
                           '(-1j*(dot(epsilonr, E))*freq*2*pi*e0)[2]', 
                           ['epsilonr', 'E'])
            # Ji : induced current = sigma *E
            add_expression(v, 'Jix', suffix, ind_vars,
                           '(dot(sigma, E))[0]', 
                           ['sigma', 'E'])
            add_expression(v, 'Jiy', suffix, ind_vars,
                           '(dot(sigma, E))[1]', 
                           ['sigma', 'E'])
            add_expression(v, 'Jiz', suffix, ind_vars,
                           '(dot(sigma, E))[2]', 
                           ['sigma', 'E'])
            # Jp : polarization current (Jp = -i omega* e0 (er - 1) E
            add_expression(v, 'Jpx', suffix, ind_vars,
                           '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[0]', 
                           ['epsilonr', 'E'])
            add_expression(v, 'Jpy', suffix, ind_vars,
                           '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[1]', 
                           ['epsilonr', 'E'])
            add_expression(v, 'Jpz', suffix, ind_vars,
                           '(-1j*(dot(epsilonr, E) - E)*freq*2*pi*e0)[2]', 
                           ['epsilonr', 'E'])
            
            
        elif name.startswith('psi'):
            add_scalar(v, 'psi', suffix, ind_vars, solr, soli)

        return v

               
    def get_fes_for_dep(self, unknown_name, soldict):
        keys = soldict.keys()
        for k in keys:
            if unknown_name.startswith('psi'):
               if k.startswith('psi'): break
            else:
               if k.startswith('E'): break
        sol = soldict[k]
        solr = sol[0]
        soli = sol[1] if len(sol) > 1 else None
        return solr, soli                
