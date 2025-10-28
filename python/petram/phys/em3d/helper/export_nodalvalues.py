'''
  export_noralvalues.py

  a helper routine to export nordalvalues from  
  a pair of solution and mesh file.

  perform 3D interpolation of using GridData

  sample usage:

     #  go to Petra-M solution directory

     >>>  import petram.helper.export_nodalvalues as exporter
     >>>  solset = exporter.load_sol(fesvar = 'E')
     >>>  values = exporter.get_nodalvalues(solset)

     >>>  R =   np.arange(0.1, 1.5, 100)
     >>>  Z =   np.arange(-1.,  1., 100)
     >>>  Phi = np.arange(0, 2*np.pi., 100)

  sample script

     x = np.linspace(0, 0.007, 5)
     y = np.linspace(0, 0.2, 50)
     z = np.linspace(0, 0.06, 10)
     X, Y, Z = np.meshgrid(x, y, z)

     #save E
     exporter.export_interpolated_data(path, X, Y, Z, 'E',
                          # vdim = 3, complex = True,
                          nproc = 1, ncfile = 'data.nc')
     #save E and B
     exporter.export_interpolated_data2(path, X, Y, Z, 
                                        1e8, 'E',
                                        nproc = 1, ncfile = 'data.nc')




  Author: S. Shiraiwa

'''
import os
import six
import numpy as np
import time
from time import gmtime, strftime

import mfem.ser as mfem
import petram

def load_sol(solfiles, fesvar, refine = 0, ):
    '''
    read sol files in directory path. 
    it reads only a file for the certain FES variable given by fesvar
    '''
    from petram.sol.solsets import find_solfiles, MeshDict
#    solfiles = find_solfiles(path)

    def fname2idx(t):    
           i = int(os.path.basename(t).split('.')[0].split('_')[-1])
           return i
    solfiles = solfiles.set
    solset = []

    for meshes, solf, in solfiles:
        s = {}
        for key in six.iterkeys(solf):
            name = key.split('_')[0]
            if name != fesvar: continue
            idx_to_read = int(key.split('_')[1])
            break

    for meshes, solf, in solfiles:
        idx = [fname2idx(x) for x in meshes]
        meshes = {i:  mfem.Mesh(str(x), 1, refine) for i, x in zip(idx, meshes) if i == idx_to_read}
        meshes=MeshDict(meshes) # to make dict weakref-able
        ### what is this refine = 0 !?
        for i in meshes.keys():
            meshes[i].ReorientTetMesh()
            meshes[i]._emesh_idx = i

        s = {}
        for key in six.iterkeys(solf):
            name = key.split('_')[0]
            if name != fesvar: continue

            fr, fi =  solf[key]
            i = fname2idx(fr)
            m = meshes[i]
            solr = (mfem.GridFunction(m, str(fr)) if fr is not None else None)
            soli = (mfem.GridFunction(m, str(fi)) if fi is not None else None)
            if solr is not None: solr._emesh_idx = i
            if soli is not None: soli._emesh_idx = i
            s[name] = (solr, soli)
        solset.append((meshes, s))
        
    return solset

def get_nodalvalues(solset,
                    curl = False,
                    grad = False,
                    div  = False):

    if grad:
        assert False, "evaluating Grad is not implemented"
    if div:
        assert False, "evaluating Div is not implemented"

    import petram.helper.eval_deriv as eval_deriv

    from collections import defaultdict
    nodalvalues =defaultdict(list)

    for meshes, s in solset:   # this is MPI rank loop
        for name in s:
            gfr, gfi = s[name]
            m = gfr.FESpace().GetMesh()
            size = m.GetNV()

            ptx = np.vstack([m.GetVertexArray(i) for i in range(size)])
            
            gfro, gfio = gfr, gfi
            if curl:
                gfr, gfi, extra = eval_deriv.eval_curl(gfr, gfi)            
            dim = gfr.VectorDim()
            
            ret = np.zeros((size, dim), dtype = float)            
            for comp in range(dim):
                values = mfem.Vector()
                gfr.GetNodalValues(values, comp+1)
                ret[:, comp] = values.GetDataArray()
                values.StealData()

            if gfi is None:
                nodalvalues[name].append((ptx, ret, gfr))
                continue

            ret2 = np.zeros((size, dim), dtype = float)                        
            for comp in range(dim):
                values = mfem.Vector()
                gfi.GetNodalValues(values, comp+1)

                if ret2 is None:
                    ret2 = np.zeros((values.Size(), dim), dtype = float)

                ret2[:, comp] = values.GetDataArray()
                values.StealData()

            ret = ret + 1j*ret2
            nodalvalues[name].append((ptx, ret, gfro))

            
    nodalvalues.default_factory = None
    return nodalvalues

def make_mask(values, X, Y, Z, mask_start = 0, logfile = None):
    '''
    mask for interpolation
    '''
    mask= np.zeros(len(X.flatten()), dtype=int) - 1

    for kk, data in enumerate(values):
       ptx, ret, gfr = data

       xmax = np.max(ptx[:, 0]);xmin = np.min(ptx[:, 0])
       ymax = np.max(ptx[:, 1]);ymin = np.min(ptx[:, 1])
       zmax = np.max(ptx[:, 2]);zmin = np.min(ptx[:, 2])

       i1 = np.logical_and(xmax >= X.flatten(), xmin <= X.flatten())
       i2 = np.logical_and(ymax >= Y.flatten(), ymin <= Y.flatten())
       i3 = np.logical_and(zmax >= Z.flatten(), zmin <= Z.flatten())
       ii = np.logical_and(np.logical_and(i1, i2), i3)

       mesh = gfr.FESpace().GetMesh()

       XX = X.flatten()[ii]
       YY = Y.flatten()[ii]
       ZZ = Z.flatten()[ii]
       size = len(XX)

       m = mfem.DenseMatrix(3, size)       
       ptx = np.vstack([XX, YY, ZZ])

       if logfile is not None:
          txt = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
          logfile.write(txt + "\n")
          logfile.write("calling FindPoints : size = "  + str(XX.shape) + " " +
                        str(kk+1) + '/' + str(len(values)) + "\n")
       
       m.Assign(ptx)
       ips = mfem.IntegrationPointArray()
       elem_ids = mfem.intArray()
       
       pts_found = mesh.FindPoints(m, elem_ids, ips)

       ii = np.where(ii)[0][np.array(elem_ids.ToList()) !=-1 ]
       mask[ii] = kk + mask_start
       if logfile is not None:
           logfile.write("done\n")
            
    return mask


def interp3D(values, X, Y, Z, mask, vdim=1, complex = False, mask_start=0,
             logfile = None):
    from scipy.interpolate import griddata
    
    X = X.flatten()
    Y = Y.flatten()
    Z = Z.flatten()        
    
    size = len(X.flatten())
    if complex:
        res = np.zeros((vdim, size), dtype=np.complex128)
    else:
        res = np.zeros((vdim, size), dtype=np.float64)
    
    for kk, data in enumerate(values):

        if logfile is not None:
            txt = strftime("%a, %d %b %Y %H:%M:%S +0000", gmtime())
            logfile.write(txt + "\n")
            logfile.write("processing interp3D :"  + str(kk+1) + '/' + str(len(values))+ "\n")
        
        idx = mask == kk + mask_start
        ptx, ret, gfr = data

        for ii in range(vdim):
           print(ptx.shape, ret.shape)
           res[ii, idx] = griddata(ptx, ret[:, ii].flatten(),
                                   (X[idx], Y[idx], Z[idx]))
           
        if logfile is not None:
            logfile.write("done\n")

    if vdim == 1: res = res.flatten()       
    return res

import multiprocessing as mp

class exporter_child(mp.Process):
    def __init__(self, result_queue, myid, rank, *args, **kwargs):

        mp.Process.__init__(self)
        self.result_queue = result_queue
        self.myid = myid
        self.rank = rank
        self.logfile = open('export.log.'+str(myid), 'w', buffering=0)
        self.args = args
        self.call_curl = kwargs.pop('curl', False)
        
 
    def data_partition(self, m, num_proc, myid):
        min_nrows  = m / num_proc
        extra_rows = m % num_proc
        start_row  = min_nrows * myid + (extra_rows if extra_rows < myid else myid)
        end_row    = start_row + min_nrows + (1 if extra_rows > myid else 0)
        nrows   = end_row - start_row
        return start_row, end_row
        
    def run(self):
        try:
            path, X, Y, Z, fesvar, vdim, complex = self.args
        
            from petram.sol.solsets import find_solfiles, MeshDict
            solfiles = find_solfiles(path)
            st, et = self.data_partition(len(solfiles.set), self.rank, self.myid)
            self.logfile.write('partition '+str(st)+':'+str(et)+'\n')
         
            solset = load_sol(solfiles[st:et], fesvar)
            self.logfile.write('calling nodalvalues\n')
            values = get_nodalvalues(solset, curl = self.call_curl)
            self.logfile.write('calling make_mask\n')            
            mask   = make_mask(values[fesvar], X, Y, Z, mask_start = st,
                               logfile = self.logfile)
            self.logfile.write('calling interp3D\n')                        
            data = interp3D(values[fesvar], X, Y, Z, mask, vdim=vdim,
                            complex=complex, mask_start = st, logfile = self.logfile)
            
            idx = np.where(mask != -1)[0]
            if vdim == 1:
                data = data[idx]
            else:
                data = data[:, idx]                
            #self.result_queue.put((mask, data))
            self.result_queue.put((idx, mask[idx],  data))
        except:
            import traceback
            self.result_queue.put((None, None, traceback.format_exc()))
        self.logfile.close()

        
def export_interpolated_data(path, X, Y, Z, fesvar,
                             vdim = 1, complex = False, nproc = 1,
                             ncfile = 'data.nc',
                             curl = False, return_mask = False):
    
    from netCDF4 import Dataset
    
    results= mp.JoinableQueue() 
    workers = [None]*nproc
    
    for i in range(nproc):
         w = exporter_child(results, i, nproc,
                            path, X, Y, Z, fesvar, vdim, complex,
                            curl=curl)
         workers[i] = w
         time.sleep(0.1)
    for w in workers:
         w.daemon = True
         w.start()
         
    res = [results.get() for x in range(len(workers))]
    for x in range(len(workers)):
         results.task_done()
         
    size = len(X.flatten())
    if complex:
        ans = np.zeros((vdim, size), dtype=np.complex128)
    else:
        ans = np.zeros((vdim, size), dtype=np.float64)
        
    mask= np.zeros(len(X.flatten()), dtype=int) - 1
    for idx, mm, dd in res:
        if mm is None:
            print(dd)
            assert False, "Child Process Failed"
        else:
            if idx.size == 0: continue
            print("here", idx.shape, dd.shape)
            if vdim == 1:
                ans[idx] = dd
            else:
                ans[:,idx] = dd
            mask[idx] = mm

    ans = ans.reshape(-1, X.shape[0], X.shape[1], X.shape[2])
    mask = mask.reshape(X.shape[0], X.shape[1], X.shape[2])    

    if ncfile != '':
        nc = Dataset(ncfile, "w", format='NETCDF4')
        nc.createDimension('vdim', vdim)
        nc.createDimension('dim_0', X.shape[0])
        nc.createDimension('dim_1', X.shape[1])
        nc.createDimension('dim_2', X.shape[2])
        if complex:        
            a_real = nc.createVariable(fesvar+'_real', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_real[:] = ans.real

            a_imag = nc.createVariable(fesvar+'_imag', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_imag[:] = ans.imag
        else:
            a_real = nc.createVariable(fesvar, np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_real[:] = ans
            
        xx = nc.createVariable('X', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))
        yy = nc.createVariable('Y', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))
        zz = nc.createVariable('Z', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))
        rank = nc.createVariable('rank', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))

        xx[:] = X
        yy[:] = Y
        zz[:] = Z
        rank[:] = mask

        nc.close()

    if return_mask:
        return ans, mask
    else:
        return ans


def export_interpolated_data2(path, X, Y, Z, freq, fesvar='E',
                              vdim = 3, complex = True,                              
                              nproc = 8,  ncfile= 'EBdata.nc'):

    from netCDF4 import Dataset
    
    Edata, mask = export_interpolated_data(path, X, Y, Z, fesvar,
                                           vdim = vdim, complex = True,
                                           nproc = nproc, return_mask = True)
    Bdata, mask = export_interpolated_data(path, X, Y, Z, fesvar,
                                           vdim = vdim, complex = True, curl=True,
                                           nproc = nproc, return_mask = True)
    

    fesvar = ''.join([x for x in fesvar if not x.isdigit()])
    omega = 2*np.pi*freq
    if ncfile != '':
        nc = Dataset(ncfile, "w", format='NETCDF4')
        nc.createDimension('vdim', vdim)
        nc.createDimension('dim_0', X.shape[0])
        nc.createDimension('dim_1', X.shape[1])
        nc.createDimension('dim_2', X.shape[2])
        if complex:        
            a_real = nc.createVariable(fesvar+'_real', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_real[:] = Edata.real

            a_imag = nc.createVariable(fesvar+'_imag', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_imag[:] = Edata.imag
            
            a_real = nc.createVariable('B_real', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_real[:] = (Bdata/1j).real/1/omega

            a_imag = nc.createVariable('B_imag', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_imag[:] = (Bdata/1j).imag/omega
            
        else:
            a_real = nc.createVariable(fesvar, np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_real[:] = Edata
            a_real = nc.createVariable('B', np.dtype('double'),
                                   ('vdim', 'dim_0', 'dim_1', 'dim_2'))
            a_real[:] = Bdata/omega
            
        xx = nc.createVariable('X', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))
        yy = nc.createVariable('Y', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))
        zz = nc.createVariable('Z', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))
        rank = nc.createVariable('rank', np.dtype('double'),
                               ('dim_0', 'dim_1', 'dim_2'))

        xx[:] = X
        yy[:] = Y
        zz[:] = Z
        rank[:] = mask

        nc.close()

'''
sample usage

x = np.linspace(0, 0.007, 5)
y = np.linspace(0, 0.2, 50)
z = np.linspace(0, 0.06, 10)
X, Y, Z = np.meshgrid(x, y, z)
data = exporter.export_interpolated_data(path, X, Y, Z, 'E',
                          vdim = 3, complex = True,
                          nproc = 1, ncfile = 'data.nc')

'''



              
