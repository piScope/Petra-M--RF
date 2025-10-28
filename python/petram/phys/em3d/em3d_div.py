'''
   divergence = 0 constraint

   this constraints add weak term, (grad u,  epsilon E), to the weak formulation.
   u is Lagurange multiplier, and descritized by H1 element.  

'''
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix, csc_matrix


from petram.model import Domain, Bdry, Pair
from petram.phys.phys_model  import Phys, VectorPhysCoefficient
import numpy as np

import petram.debug as debug
#from petram.helper.chypre_to_pymatrix import *
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Div')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
else:
   import mfem.ser as mfem

#from mfem.common.sparse_utils import eliminate_rows, eliminate_cols , sparsemat_to_scipycsr
class Arctan(mfem.PyCoefficient):
   def EvalValue(self, x):
       return  np.arctan2(x[0], x[2])

def domain_constraints():
   return [EM3D_Div]
    
class EM3D_Div(Domain, Phys):
    is_secondary_condition = True   
    has_essential = True
    def __init__(self, **kwargs):
        super(EM3D_Div, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        super(EM3D_Div, self).attribute_set(v)
        v['sel_readonly'] = False
        v['sel_index'] = []
        v['psi0'] = 1.0
        v['essential_bdr'] = ''
        return v
    
    def panel1_param(self):
        return [["psi",  float(self.psi0),  300, {}],
                ["essential_bdr",  self.essential_bdr,  0, {}],]
                
    def get_panel1_value(self):
        return (float(self.psi0),
                self.essential_bdr)

    def import_panel1_value(self, v):
        self.psi0 = float(v[0])
        self.essential_bdr = str(v[1])

       #def has_extra_DoF(self):
    #    return True

    def get_essential_idx(self, idx):
        if idx == 0: 
            return []
        else:
            if self.essential_bdr.strip() == '': return []
            bdr = [int(x) for x in self.essential_bdr.strip().split(',')]
            return bdr

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag of MixedBilinearForm
           flag1 : take transpose
           flag2 : take conj
        '''
        return [(1, 0, -1, -1), (1, 0, 1, 1)]

    ''' 
    def postprocess_extra(self, sol, flag, sol_extra):
        name = 'psi_div'
        sol_extra[name] = np.array(sol)
    '''

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        dprint1("Add Mixed contribution: " + str((r, c, is_trans, real)))
        itg =  mfem.MixedVectorWeakDivergenceIntegrator
        domains = [engine.find_domain_by_index(self.get_root_phys(), x,
                                               check_enabled = True)
                   for x in self._sel_index]
        
        self.set_integrator_realimag_mode(real)        
        for dom, idx in zip(domains, self._sel_index):
            coeff1, coeff2, coeff3 = dom.get_coeffs()
            
            if dom.has_pml():
               coeff1 = dom.make_PML_epsilon(coeff1)
               coeff3 = dom.make_PML_sigma(coeff3)
           
            self.add_integrator(engine, 'epsilonr', coeff1,
                                mbf.AddDomainIntegrator,
                                itg, idx=[idx], vt  = dom.vt, transpose=is_trans)
            self.add_integrator(engine, 'sigma', coeff3,
                                mbf.AddDomainIntegrator,
                                itg, idx=[idx], vt  = dom.vt, transpose=is_trans)
            
    def apply_essential(self, engine, gf, real = False, kfes = 0):
        if kfes == 0: return
        if real:       
            dprint1("Apply Ess.(real)" + str(self._sel_index))
            coeff = mfem.ConstantCoefficient(float(self.psi0))
        else:
            return
            #dprint1("Apply Ess.(imag)" + str(self._sel_index))
            #coeff = mfem.ConstantCoefficient(complex(self.psi0).imag)

        coeff = self.restrict_coeff(coeff, engine)
        mesh = engine.get_mesh(mm = self)        
        ibdr = mesh.bdr_attributes.ToList()
        bdr_attr = [0]*mesh.bdr_attributes.Max()

        ess_bdr = self.get_essential_idx(1)      
        dprint1("Essential boundaries", ess_bdr)
 
        for idx in ess_bdr:
            bdr_attr[idx-1] = 1
        gf.ProjectBdrCoefficient(coeff, mfem.intArray(bdr_attr))

    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        '''
        no dialoganl element except for essentail DoF
        '''
        pass
        
