import numpy as np

from petram.mfem_config import use_parallel
import petram.debug

dprint1, dprint2, dprint3 = petram.debug.init_dprints('eval_deriv')

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
    l2_coll = mfem.L2_FECollection(p - 1, sdim)

    rts = FiniteElementSpace(mesh, l2_coll, vdim, ordering)

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
