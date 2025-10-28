from __future__ import print_function

import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix

from petram.model import Pair, Bdry
from petram.phys.phys_model  import Phys

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_Floquet')

from petram.mfem_config import use_parallel

if use_parallel:
   from mfem.common.parcsr_extra import ToHypreParCSR, get_row_partitioning
   import mfem.par as mfem
   from mpi4py import MPI
   num_proc = MPI.COMM_WORLD.size
   myid     = MPI.COMM_WORLD.rank
   from mfem.common.mpi_debug import nicePrint
else:
   import mfem.ser as mfem
   num_proc = 1
   myid = 0

'''
   Map DoF from src surface (s1) to dst surface (s2)
   
   s1 and s2 should be plain.

   axis of rotation from s1 to s2 should be perpdicular to
   normal vectors of s1 and s2.

   Twist is not considered?

   For a fineite (non 0, non 180) complex phase difference
   compulex conjugate is returned, to force Lagrange multiplier
   real
'''
def pair_constraints():
   return [EM3D_Floquet]

class EM3D_Floquet(Pair,Bdry, Phys):
    def __init__(self,  **kwargs):
        super(EM3D_Floquet, self).__init__(**kwargs)
        Phys.__init__(self)
        
    def attribute_set(self, v):
        v['dphase'] = 0.0
        v['dphase_txt'] = '0.0'
        v['map_to_u'] = "x"
        v['map_to_v'] = "y"
        v['tol_txt'] = "1e-4"
        v['tol'] = 1e-4
        v['use_multiplier'] = False
        super(EM3D_Floquet, self).attribute_set(v)
        return v
        
    def panel1_param(self):
        return [['u mapping ',  self.map_to_u, 0, {}],
                ['v mapping ',  self.map_to_v, 0,  {}],
                self.make_phys_param_panel('phase(deg)',  self.dphase_txt),
                self.make_phys_param_panel('tol.',  self.tol_txt),
                ["", "E_dst = exp(i*angle)*E_src" ,2, None],]     
#                ["use Lagrange multiplier",   self.use_multiplier,  3, {"text":""}],]     

    def get_panel1_value(self):
        return (self.map_to_u, self.map_to_v, self.dphase_txt,
                self.tol_txt, None, self.use_multiplier)

    def import_panel1_value(self, v):
        self.map_to_u = v[0]
        self.map_to_v = v[1]
        self.dphase_txt  = str(v[2])
        self.tol_txt  = str(v[3])
        #self.use_multiplier  = bool(v[4])
        self.use_multiplier  = False
        
    def preprocess_params(self, engine):
        ### find normal (outward) vector...
        map_to_u = ['def map_to_u(xyz):',
                    '    import numpy as np',
                    '    x = xyz[0]; y=xyz[1]; z=xyz[2]',
                    '    return '+self.map_to_u]
        map_to_v = ['def map_to_v(xyz):',
                    '    import numpy as np',
                    '    x = xyz[0]; y=xyz[1]; z=xyz[2]',
                    '    return '+self.map_to_v]

        self.func_u = map_to_u
        self.func_v = map_to_v

        dprint1("u(x, y, z)")
        dprint1(self.func_u)
        dprint1("v(x, y, z)") 
        dprint1(self.func_v)
        try:
            exec('\n'.join(self.func_u))
            exec('\n'.join(self.func_v))
        except:
            raise ValueError("Cannot complie mappling rule")

        try:
            self.dphase = float(self.eval_phys_expr(str(self.dphase_txt), 'dphase')[0])
            self.tol = float(self.eval_phys_expr(str(self.tol_txt), 'tol')[0])
        except:
            import traceback
            traceback.print_exc()
            raise ValueError("Cannot evaluate dphase/tolerance to float number")

    def has_extra_DoF(self, kfes = 0):
        return False

    def has_interpolation_contribution(self, kfes = 0):
        #if kfes != 0: return False
        if self.use_multiplier: return False
        return True

    def add_interpolation_contribution(self, engine, ess_tdof=None, kfes = 0):
        dprint1("Add interpolation contribution(real)" + str(self._sel_index))
        dprint1("kfes = ", str(kfes))        
        dprint1("dphase = ", str(self.dphase))        



        trans1= ['def trans1(xyz):',
                 '    import numpy as np',
                 '    x = xyz[0]; y=xyz[1]; z=xyz[2]',
                 '    return np.array(['+self.map_to_u + ','+ self.map_to_v+'])']
        
        lns = {}
        exec('\n'.join(self.func_u), self._global_ns, lns) # this defines map_to_u
        exec('\n'.join(self.func_v), self._global_ns, lns) # this defines map_to_v
        exec('\n'.join(trans1), self._global_ns, lns)      # this defines trans1
        
        map_to_u = lns['map_to_u']
        map_to_v = lns['map_to_v']
        trans1 = lns['trans1']
        
        tdof = [] if ess_tdof is None else ess_tdof
        src = self._src_index
        dst = [x for x in self._sel_index if not x in self._src_index]

        fes = engine.get_fes(self.get_root_phys(), kfes)

        #old version
        '''
        from petram.helper.dof_mapping_matrix import dof_mapping_matrix
        M, r, c = dof_mapping_matrix(src,  dst,  fes, ess_tdof, 
                                     engine, self.dphase,
                                     map_to_u = map_to_u, map_to_v = map_to_v,
                                     smap_to_u = map_to_u, smap_to_v = map_to_v,
                                     tol = self.tol)
        #nicePrint(M.GetRowPartArray())
        #nicePrint(M.GetColPartArray())
        #nicePrint(M.shape)
        '''
        from petram.helper.dof_map import projection_matrix
        M, r, c = projection_matrix(src, dst, fes, ess_tdof, fes2=fes,
                                    trans1 = trans1, dphase = self.dphase,
                                    tol = self.tol, mode = 'surface')
        return M, r, c
        
    '''
    def add_extra_contribution(self, engine, ess_tdof=None, **kwargs):
        dprint1("Add Extra contribution(real)" + str(self._sel_index))        

        from petram.helper.find_dof_map2 import find_dof_map2

        lns = {}        
        exec '\n'.join(self.func_u) in self._global_ns, lns # this defines map_to_u
        exec '\n'.join(self.func_v) in self._global_ns, lns # this defines map_to_v
        map_to_u = lns['map_to_u']
        map_to_v = lns['map_to_v']

        tdof = [] if ess_tdof is None else ess_tdof
        src = self._src_index
        dst = [x for x in self._sel_index if not x in self._src_index]
        
        map1 = find_dof_map2(src, dst,
                             map_to_u, map_to_v, tdof, engine, self.tol,
                             use_complex=False)
        map2 = map1.tocsr()
        map1 = map1.astype(complex)        
        num_nonzeros = np.diff(map2.indptr)
        k_nonzeros = np.where(num_nonzeros != 0)[0]
        #map1 column continous....and contains t2 (horizontal)

        # fill diagnal element
        if use_parallel:
            partitioning = get_row_partitioning(engine.r_A)
            start_row = partitioning[myid]
            end_row = partitioning[myid+1]
            k_nonzeros =  engine.allgather_vector(k_nonzeros)
            for i in k_nonzeros:
               if i >= start_row and i < end_row:
                    map1[i, i] = -np.exp(-1j*np.pi/180*self.dphase)

        else:
            start_row = 0
            end_row = map1.shape[1]
            for i in k_nonzeros:
                 map1[i, i] = -np.exp(-1j*np.pi/180*self.dphase)
            

        
        # set matrix
        # note that zero row/col are not eliminated here.
        # it will be eliminated in engine after independent variables
        # are founds, after evaulating interpolation matrix)
        
        if use_parallel:
            nrows = end_row - start_row             
            map1 = map1.transpose()

            m1 = csr_matrix(map1.real[start_row:end_row, :], dtype=float)
            m2 = csr_matrix(map1.imag[start_row:end_row, :], dtype=float)
            from mfem.common.chypre import CHypreMat
            M1 = CHypreMat(m1, m2).conj()
            M2 = M1.transpose().conj()
            dprint1('M1 Shape', M1.shape, ' NNZ ', M1.nnz)
            dprint1('M2 Shape', M2.shape, ' NNZ ', M2.nnz)            

            return (M1, M2, None, 0, False)
        else:
            map1 = lil_matrix(map1, dtype=complex)
            return (map1.transpose().conj(), map1, None, 0, False)
    '''



    
