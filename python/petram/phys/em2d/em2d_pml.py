'''
   PML

   * Linear PML
            [S(x)          ]
        S = [     S(y)     ]

        S  = 1 + (Sr + iSi)*((r -Lr)/Lpml)^p

        Lr = the coordinate of boundar between normal and PML
        Lpml = PML thickness
        Sr, Si = PML strech factor
        p = the order of PML stretch

      We replase epsilon and mu in the PML region as follow
         epsilonr -> G * epsilon * S^-1, 
         mur -> G * mur * S^-1,
      where G = det(S) S^-1. det(S) is S(x)*S(y)*S(z) for linear case

      CopyRight (c) 2020-  S. Shiraiwa
'''
from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.pycomplex_coefficient import CC_Matrix
from petram.mfem_config import use_parallel
import numpy as np
import abc
from abc import ABC, abstractmethod

from petram.phys.phys_model import Phys, PhysCoefficient, MatrixPhysCoefficient
from petram.phys.em2d.em2d_base import EM2D_Bdry, EM2D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2D_PML')

if use_parallel:
    import mfem.par as mfem
else:
    import mfem.ser as mfem

data = (('stretch', VtableElement('stretch', type='complex',
                                  guilabel='stretch',
                                  default=1.0,
                                  no_func=True,
                                  tip="real: shorten wavelength (>0), imag: damping (>0 forward, <0 backward)")),
        ('s_order', VtableElement('s_order', type='float',
                                  guilabel='order',
                                  default=1.0,
                                  no_func=True,
                                  tip="streach order")),)


class EM2D_PML(EM2D_Domain, ABC):
    @abstractmethod
    def make_PML_coeff(self):
        pass

    def get_parent_selection(self):
        return self.parent.sel_index


class LinearPML(CC_Matrix):
    def __init__(self, coeff, S, direction, ref_point, pml_width, order, inv):
        self.coeffs = coeff
        self.direction = direction
        self.ref_p = ref_point
        self.pml_width = pml_width
        self.S = S
        self.order = order
        self.return_inv = inv
        self.coeff = coeff
        super(LinearPML, self).__init__(3, 3)

    def eval(self, T, ip):
        ptx = T.Transform(ip)

        invS = self.Eval_invS(ptx)
        detS = self.Eval_detS(ptx)

        K_m = self.coeffs.eval(T, ip)

        detS_inv_S_x_inv_S = (invS*detS).dot(K_m).dot(invS)

        if self.return_inv:
            detS_inv_S_x_inv_S = np.linalg.inv(detS_inv_S_x_inv_S)

        return detS_inv_S_x_inv_S

    def Eval_S(self, x):
        ret = np.array([1+0j, 1., 1.])

        if self.direction[0]:
            ret[0] = 1 + self.S * \
                ((x[0] - self.ref_p[0])/self.pml_width[0])**self.order
        if self.direction[1]:
            ret[1] = 1 + self.S * \
                ((x[1] - self.ref_p[1])/self.pml_width[1])**self.order
        return ret

    def Eval_invS(self, x):
        ret = self.Eval_S(x)
        return np.diag((1+0j)/ret)

    def Eval_detS(self, x):
        ret = self.Eval_S(x)
        return ret[0]*ret[1]

def linear_pml_func1(ptx, K_m):
    Sd = np.array([1+0j, 1., 1.])

    if direction[0]:
        Sd[0] = 1 + S * ((ptx[0] - ref_p[0])/pml_width[0])**pml_order
    if direction[1]:
        Sd[1] = 1 + S * ((ptx[1] - ref_p[1])/pml_width[1])**pml_order

    invS = np.diag((1+0j)/Sd)
    detS = Sd[0]*Sd[1]
    K_m = np.ascontiguousarray(K_m)

    detS_inv_S_x_inv_S = (invS*detS).dot(K_m).dot(invS)

    return detS_inv_S_x_inv_S

def linear_pml_func2(ptx):
    Sd = np.array([1+0j, 1., 1.])

    if direction[0]:
        Sd[0] = 1 + S * ((ptx[0] - ref_p[0])/pml_width[0])**pml_order
    if direction[1]:
        Sd[1] = 1 + S * ((ptx[1] - ref_p[1])/pml_width[1])**pml_order

    invS = np.diag((1+0j)/Sd)
    detS = Sd[0]*Sd[1]

    detS_inv_S_x_inv_S = (invS*detS).dot(K_m).dot(invS)

    return detS_inv_S_x_inv_S


class EM2D_LinearPML(EM2D_PML):
    has_2nd_panel = False
    has_3rd_panel = False
    _has_4th_panel = False
    vt = Vtable(data)

    def attribute_set(self, v):
        v = super(EM2D_LinearPML, self).attribute_set(v)
        v["stretch_dir"] = [False, False]
        v["ref_point"] = ""
        return v

    def panel1_param(self):
        panels1 = [["Direction", self.stretch_dir, 36,
                    {'col': 3, 'labels': ['x', 'y']}],
                   ["Ref. point", self.ref_point, 0, {}],
                   ]
        panels2 = super(EM2D_LinearPML, self).panel1_param()

        panels = panels1 + panels2
        return panels

    def get_panel1_value(self):
        val = super(EM2D_LinearPML, self).get_panel1_value()
        val = [self.stretch_dir, self.ref_point] + val
        return val

    def import_panel1_value(self, v):
        super(EM2D_LinearPML, self).import_panel1_value(v[2:])
        self.ref_point = str(v[1])
        self.stretch_dir = [vv[1] for vv in v[0]]

    def panel1_tip(self):
        tip1 = ["PML direction", "reference point to measure the PML thickness"]
        tip2 = super(EM2D_LinearPML, self).panel1_tip()
        return tip1 + tip2

    def preprocess_params(self, engine):
        super(EM2D_LinearPML, self).preprocess_params(engine)
        dprint1("PML (stretch, order) ", self.stretch, self.s_order)

        base_mesh = engine.meshes[0]

        ec = base_mesh.extended_connectivity
        v2v = ec['vert2vert']

        idx = int(self.ref_point)
        self.ref_point_coord = base_mesh.GetVertexArray(v2v[idx]).copy()

        dprint1("PML reference point:", self.ref_point_coord)

        l = sum([ec['surf2line'][k] for k in self.get_parent_selection()], [])
        l = list(set(l))
        v = sum([ec['line2vert'][k] for k in l], [])
        v = list(set(v))

        ptx = np.vstack([base_mesh.GetVertexArray(v2v[idx]) for idx in v])

        self.pml_width = [0]*2
        for k in range(2):
            dd = ptx[:, k] - self.ref_point_coord[k]
            ii = np.argmax(np.abs(dd))
            self.pml_width[k] = dd[ii]

        dprint1("PML width ", self.pml_width)

    def make_PML_coeff(self, coeff):
        from petram.phys.numba_coefficient import NumbaCoefficient
        from petram.mfem_config import numba_debug
        from petram.phys.pycomplex_coefficient import (PyComplexConstant,
                                                       PyComplexMatrixConstant,)

        if isinstance(coeff, (NumbaCoefficient,
                              PyComplexConstant,
                              PyComplexMatrixConstant)):
            params = {"S": self.stretch,
                      "direction": np.array(self.stretch_dir),
                      "ref_p": np.array(self.ref_point_coord),
                      "pml_width": np.array(self.pml_width),
                      "pml_order": self.s_order, }

            if isinstance(coeff, NumbaCoefficient):
                dep = (coeff.mfem_numba_coeff,)
                coeff = mfem.jit.matrix(sdim=2,
                                        shape=(3, 3),
                                        complex=True,
                                        debug=numba_debug,
                                        dependency=dep,
                                        params=params,
                                        interface="simple")(linear_pml_func1)
            else:
                if isinstance(coeff, PyComplexConstant):
                    value = np.diag([coeff.value]*3)
                else:
                    value = coeff.value
                params["K_m"] = value
                coeff = mfem.jit.matrix(sdim=2,
                                        shape=(3, 3),
                                        complex=True,
                                        debug=numba_debug,
                                        params=params,
                                        interface="simple")(linear_pml_func2)

            coeff = NumbaCoefficient(coeff)
        else:
            coeff = LinearPML(coeff, self.stretch, self.stretch_dir,
                              self.ref_point_coord, self.pml_width, self.s_order,
                              False)
        return coeff
    '''
    def make_PML_invmu(self):
        coeff = LinearPML(coeff, self.stretch, self.stretch_dir,
                          self.ref_point_coord, self.pml_width, self.s_order,
                          True)
        return coeff

    def make_PML_sigma(self):
        coeff = LinearPML(coeff, self.stretch, self.stretch_dir,
                          self.ref_point_coord, self.pml_width, self.s_order,
                          False)
        return coeff
    '''
