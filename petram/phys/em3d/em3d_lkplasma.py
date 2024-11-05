'''
   local-K plasma.
'''
from petram.phys.common.rf_dispersion_lkplasma import (vtable_data0,
                                                       default_kpe_option,
                                                       kpe_options)

import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.vtable import VtableElement, Vtable
from petram.phys.em3d.em3d_const import mu0, epsilon0
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff
from petram.phys.numba_coefficient import NumbaCoefficient

from petram.phys.phys_model import MatrixPhysCoefficient, PhysCoefficient, PhysConstant, PhysMatrixConstant
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_LocalKPlasma')

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

vtable_data = vtable_data0.copy()

kpe_alg_options = ["std", ]


def domain_constraints():
    return [EM3D_LocalKPlasma]


class EM3D_LocalKPlasma(EM3D_Domain):
    allow_custom_intorder = True
    vt = Vtable(vtable_data)
    # nlterms = ['epsilonr']

    def get_possible_child(self):
        from .em3d_pml import EM3D_LinearPML
        return [EM3D_LinearPML]

    def attribute_set(self, v):
        super(EM3D_LocalKPlasma, self).attribute_set(v)
        v["kpe_mode"] = default_kpe_option
        v["kpe_alg"] = kpe_alg_options[0]
        return v

    def panel1_param(self):
        panels = super(EM3D_LocalKPlasma, self).panel1_param()
        panels.append(["kpe mode", None, 1, {"values": kpe_options}])
        panels.append(["kpe alg.", None, 1, {"values": kpe_alg_options}])

        return panels

    def get_panel1_value(self):
        values = super(EM3D_LocalKPlasma, self).get_panel1_value()
        values.extend([self.kpe_mode, self.kpe_alg])
        return values

    def import_panel1_value(self, v):
        check = super(EM3D_LocalKPlasma, self).import_panel1_value(v[:-2])
        self.kpe_mode = v[-2]
        self.kpe_alg = v[-1]
        return check

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        from .em3d_const import mu0, epsilon0
        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, t_i, t_c, masses, charges, kpakpe, kpevec = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars
        kpe_mode = self.kpe_mode
        kpe_alg = self.kpe_alg

        from petram.phys.common.rf_dispersion_lkplasma import build_coefficients
        coeff1, coeff2, coeff3, coeff4 = build_coefficients(ind_vars, omega, B, t_c, dens_e, t_e,
                                                            dens_i, t_i, masses, charges, kpakpe, kpevec,
                                                            kpe_mode, self._global_ns, self._local_ns,
                                                            kpe_alg=kpe_alg, sdim=3)

        return coeff1, coeff2, coeff3, coeff4

    def has_bf_contribution(self, kfes):
        if kfes == 0:
            return True
        else:
            return False

    def add_bf_contribution(self, engine, a, real=True, kfes=0):
        if kfes != 0:
            return
        if real:
            dprint1("Add BF contribution(real)" + str(self._sel_index))
        else:
            dprint1("Add BF contribution(imag)" + str(self._sel_index))

        coeff1, coeff2, coeff3, coeff4 = self.jited_coeff
        self.set_integrator_realimag_mode(real)

        if self.has_pml():
            coeff1 = self.make_PML_coeff(coeff1)
            coeff2 = self.make_PML_coeff(coeff2)
            coeff3 = self.make_PML_coeff(coeff3)
        coeff2 = coeff2.inv()

        if self.allow_custom_intorder and self.add_intorder != 0:
            fes = a.FESpace()
            geom = fes.GetFE(0).GetGeomType()
            order = fes.GetFE(0).GetOrder()
            isPK = (fes.GetFE(0).Space() == mfem.FunctionSpace.Pk)
            orderw = fes.GetElementTransformation(0).OrderW()
            curlcurl_order = order*2 - 2 if isPK else order*2
            mass_order = orderw + 2*order
            curlcurl_order += self.add_intorder
            mass_order += self.add_intorder

            dprint1("Debug: custom int order. Increment = " +
                    str(self.add_intorder))
            dprint1("  FE order: " + str(order))
            dprint1("  OrderW: " + str(orderw))
            dprint1("  CurlCurlOrder: " + str(curlcurl_order))
            dprint1("  FEMassOrder: " + str(mass_order))

            cc_ir = mfem.IntRules.Get(geom, curlcurl_order)
            ms_ir = mfem.IntRules.Get(geom, mass_order)
        else:
            cc_ir = None
            ms_ir = None

        self.add_integrator(engine, 'epsilonr', coeff1,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator,
                            ir=ms_ir)

        if coeff2 is not None:
            self.add_integrator(engine, 'mur', coeff2,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator,
                                ir=cc_ir)
            # coeff2 = self.restrict_coeff(coeff2, engine)
            # a.AddDomainIntegrator(mfem.CurlCurlIntegrator(coeff2))
        else:
            dprint1("No contrinbution from curlcurl")

        self.add_integrator(engine, 'sigma', coeff3,
                            a.AddDomainIntegrator,
                            mfem.VectorFEMassIntegrator,
                            ir=ms_ir)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_expression, add_constant

        if len(self._sel_index) == 0:
            return

        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, t_i, t_c, masses, charges, kpakpe, kpevec = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars
        kpe_mode = self.kpe_mode
        kpe_alg = self.kpe_alg

        from petram.phys.common.rf_dispersion_lkplasma import build_variables

        ss = self.parent.parent.name()+'_'+self.name()  # phys module name + name
        ret = build_variables(v, ss, ind_vars,
                              omega, B, t_c, dens_e, t_e,
                              dens_i, t_i, masses, charges,
                              kpakpe, kpevec, kpe_mode, kpe_alg,
                              self._global_ns, self._local_ns,
                              sdim=3)

        from petram.phys.common.rf_dispersion_lkplasma import add_domain_variables_common
        add_domain_variables_common(self, ret, v, suffix, ind_vars)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        return
