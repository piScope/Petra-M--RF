'''
   write_cdf: write netCDF file

       solset: solset
       battrs: list of boundary index

       curl : True/False if include curlE in file

       params : a list of callables
                other parameters evaluated on all x, y, z
                these should be namelist params.

   for cold plasma
        P, D, S, Br, Bz, Bt, Te, ne
        
'''   
from petram.utils import eval_sol
from petram.utils import eval_curl
from netCDF4 import Dataset
from numpy import pi, exp, abs, concatenate, dstack, array

def eval_solset(solsets, battrs, func, dim):
    ret = {}
    for battr in battrs:
        ret[battr] = []
        for m0, solr, soli in solsets:
             if soli is None:
                v,  cdata = func(solr, battr, dim)
                if v is None: continue
                ret[battr].append((v, cdata))
             else:
                v, cdata1 = func(solr, battr, dim)
                if v is None: continue
                v, cdata2 = func(soli, battr, dim)
                ret[battr].append((v, (cdata1 + 1j*cdata2)))
                #ret[battr].append((v, abs(cdata1 + 1j*cdata2)))
    return ret

def write_complex(rootgrp, name, value):
    ReE = rootgrp.createVariable("Re"+name, "f8", ("nele", "npts"))
    ImE = rootgrp.createVariable("Im"+name, "f8", ("nele", "npts"))
    ReE[:] = value.real
    ImE[:] = value.imag

def write_float(rootgrp, name, value):
    E = rootgrp.createVariable(name, "f8", ("nele", "npts"))
    E[:] = value

def create_cdf(filename, solsets, battrs):
    rootgrp = Dataset(filename, 'w', format='NETCDF3_64BIT')
    v  = eval_solset(solsets, battrs, eval_sol,  1)
    xy   = concatenate([concatenate([x[0] for x in v[i][:]]) for i in battrs])
    nele = rootgrp.createDimension('nele', xy.shape[0])
    ndim = rootgrp.createDimension('ndim', xy.shape[-1])
    npts = rootgrp.createDimension('npts', xy.shape[1])
    write_float(rootgrp, 'x', xy[:,:,0])
    write_float(rootgrp, 'y', xy[:,:,1])
    write_float(rootgrp, 'z', xy[:,:,2])
#    pts = rootgrp.createVariable("pts", "f8", ("nele", "npts", "ndim"))
#    pts[:] = xy
    rootgrp.close()

def add_sol_variable(filename, solsets, battrs, curl=False):
    rootgrp = Dataset(filename, 'a')    
    Ex  = eval_solset(solsets, battrs, eval_sol,  1)
    Ey  = eval_solset(solsets, battrs, eval_sol,  2)
    Ez  = eval_solset(solsets, battrs, eval_sol,  3)

    Exa  = concatenate([concatenate([x[1] for x in Ex[i][:]]) for i in battrs])
    Eya  = concatenate([concatenate([x[1] for x in Ey[i][:]]) for i in battrs])
    Eza  = concatenate([concatenate([x[1] for x in Ez[i][:]]) for i in battrs])

    write_complex(rootgrp, "Ex", Exa)
    write_complex(rootgrp, "Ey", Eya)
    write_complex(rootgrp, "Ez", Eza)

    if curl:
        cEx = eval_solset(solsets, battrs, eval_curl, 1)
        cEy = eval_solset(solsets, battrs, eval_curl, 2)
        cEz = eval_solset(solsets, battrs, eval_curl, 3)
        cExa = concatenate([concatenate([x[1] for x in cEx[i][:]]) for i in battrs])
        cEya = concatenate([concatenate([x[1] for x in cEy[i][:]]) for i in battrs])
        cEza = concatenate([concatenate([x[1] for x in cEz[i][:]]) for i in battrs])

        write_complex(rootgrp, "cEx", cExa)
        write_complex(rootgrp, "cEy", cEya)
        write_complex(rootgrp, "cEz", cEza)

def add_ns_variable(filename, namespace, name, complex=False):
    rootgrp = Dataset(filename, 'r')
    x = rootgrp.variables['x'][:]
    y = rootgrp.variables['y'][:]
    z = rootgrp.variables['z'][:]

    pts = dstack((x, y, z))
    rootgrp.close()
    pp = pts.reshape(-1, pts.shape[-1])
    f =  namespace[name]
    data = array([f(*p) for p in pp])
    data = data.reshape((pts.shape[0], pts.shape[1]))

    rootgrp = Dataset(filename, 'a')
    if complex:
        write_complex(rootgrp, name, data)
    else:
        write_float(rootgrp, name, data)
    rootgrp.close()

def example(solsets, ns, filename='icrf.nc', battrs = [9]):
    create_cdf('icrf.nc', solsets, battrs)
    add_sol_variable(filename, solsets, battrs, curl=True)

    add_ns_variable(filename, ns, 'P_rf', complex=True)
    add_ns_variable(filename, ns, 'S_rf', complex=True)
    add_ns_variable(filename, ns, 'D_rf', complex=True)

    add_ns_variable(filename, ns, 'ne')
    add_ns_variable(filename, ns, 'ni')
    add_ns_variable(filename, ns, 'nim')
    add_ns_variable(filename, ns, 'Te')
    add_ns_variable(filename, ns, 'Ti')
    add_ns_variable(filename, ns, 'Tim')

    add_ns_variable(filename, ns, 'Br')
    add_ns_variable(filename, ns, 'Bz')
    add_ns_variable(filename, ns, 'Bt')



   
