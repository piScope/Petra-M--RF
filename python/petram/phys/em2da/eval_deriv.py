'''

  evaulate derivative of Nedelec element using
  DiscreteLinearOperator

'''
import mfem
import numpy as np

from petram.mfem_config import use_parallel
import petram.debug

dprint1, dprint2, dprint3 = petram.debug.init_dprints('eval_deriv')

if mfem.__version__.startswith('4.2'):
    isMFEM42 = True
else:
    isMFEM42 = False

if use_parallel:
    import mfem.par as mfem
    FiniteElementSpace = mfem.ParFiniteElementSpace
    DiscreteLinearOperator = mfem.ParDiscreteLinearOperator
    GridFunction = mfem.ParGridFunction
else:
    import mfem.ser as mfem
    FiniteElementSpace = mfem.FiniteElementSpace
    DiscreteLinearOperator = mfem.DiscreteLinearOperator
    GridFunction = mfem.GridFunction


def eval_curl(gfr, gfi=None):
    '''
    evaluate curl, gfr/gfi is supposed to be Et
    '''
    fes = gfr.FESpace()
    ordering = fes.GetOrdering()
    mesh = fes.GetMesh()
    vdim = 1
    sdim = mesh.SpaceDimension()
    p = fes.GetOrder(0)
    rt_coll = mfem.L2_FECollection(p - 1, sdim)

    rts = FiniteElementSpace(mesh, rt_coll, vdim, ordering)

    curl = DiscreteLinearOperator(fes, rts)
    itp = mfem.CurlInterpolator()
    curl.AddDomainInterpolator(itp)
    curl.Assemble()
    curl.Finalize()

    br = GridFunction(rts)
    curl.Mult(gfr, br)
    if gfi is not None:
        bi = GridFunction(rts)
        curl.Mult(gfi, bi)
    else:
        bi = None

    if isMFEM42:
        # this is meant for adjustment only for MFEM4.2.
        # it is confirmed not needed in 4.3 (and probably later version of 4.2)
        arr = br.GetDataArray()
        scale = np.zeros(len(arr))

        for ii in range(rts.GetNE()):
            trans = rts.GetElementTransformation(ii)
            dofs = rts.GetElementDofs(ii)
            scale[dofs] = trans.Weight()

        brr = br.GetDataArray()
        brr /= scale
        if bi is not None:
            bii = bi.GetDataArray()
            bii /= scale

    return br, bi, rts


def eval_grad(gfr, gfi=None):
    '''
    evaluate grad
    '''
    fes = gfr.FESpace()
    ordering = fes.GetOrdering()
    mesh = fes.GetMesh()
    vdim = 1
    sdim = mesh.SpaceDimension()
    p = fes.GetOrder(0)
    #rt_coll = mfem.L2_FECollection(p - 1, sdim)
    #rt_coll = mfem.ND_FECollection(p - 1, sdim)
    rt_coll = mfem.ND_FECollection(p, sdim)

    rts = FiniteElementSpace(mesh, rt_coll, vdim, ordering)

    grad = DiscreteLinearOperator(fes, rts)
    itp = mfem.GradientInterpolator()
    grad.AddDomainInterpolator(itp)
    grad.Assemble()
    grad.Finalize()

    br = GridFunction(rts)
    grad.Mult(gfr, br)
    if gfi is not None:
        bi = GridFunction(rts)
        grad.Mult(gfi, bi)
    else:
        bi = None
    # needs to return rts to prevent rts to be collected.
    return br, bi, rts
