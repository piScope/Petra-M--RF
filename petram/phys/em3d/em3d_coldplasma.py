'''
   cold plasma.
'''
from petram.phys.common.rf_dispersion_coldplasma import (stix_options,
                                                         default_stix_option,
                                                         default_stix_modelvalue,
                                                         col_model_options,
                                                         default_col_model,
                                                         vtable_data0)

import numpy as np

from petram.mfem_config import use_parallel, get_numba_debug

from petram.phys.vtable import VtableElement, Vtable
from petram.phys.phys_const import mu0, epsilon0
from petram.phys.coefficient import SCoeff, VCoeff, MCoeff
from petram.phys.numba_coefficient import NumbaCoefficient

from petram.phys.phys_model import MatrixPhysCoefficient, PhysCoefficient, PhysConstant, PhysMatrixConstant
from petram.phys.em3d.em3d_base import EM3D_Bdry, EM3D_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_ColdPlasma')

if use_parallel:
    import mfem.par as mfem
    from mpi4py import MPI
    myid = MPI.COMM_WORLD.rank
else:
    import mfem.ser as mfem
    myid = 0

vtable_data = vtable_data0.copy()


def domain_constraints():
    return [EM3D_ColdPlasma]


class EM3D_ColdPlasma(EM3D_Domain):
    allow_custom_intorder = True
    vt = Vtable(vtable_data)
    # nlterms = ['epsilonr']

    def get_possible_child(self):
        from .em3d_pml import EM3D_LinearPML
        return [EM3D_LinearPML]

    def attribute_set(self, v):
        super(EM3D_ColdPlasma, self).attribute_set(v)
        v["stix_terms"] = default_stix_modelvalue
        v["col_model"] = default_col_model
        return v

    def config_terms(self, evt):
        from petram.phys.common.rf_stix_terms_panel import ask_rf_stix_terms

        self.vt.preprocess_params(self)
        _B, _dens_e, _t_e, _dens_i, _masses, charges = self.vt.make_value_or_expression(
            self)

        num_ions = len(charges)
        win = evt.GetEventObject()
        value = ask_rf_stix_terms(win, num_ions, self.stix_terms)
        self.stix_terms = value

    def stix_terms_str(self):
        from petram.phys.common.rf_dispersion_coldplasma import value2modelstr
        return value2modelstr(self.stix_terms)

    def panel1_param(self):
        panels = super(EM3D_ColdPlasma, self).panel1_param()
        panels.append(["col. model", None, 1, {"values": col_model_options}])
        panels.extend([["Stix terms", "", 2, None],
                       [None, None, 341, {"label": "Customize terms",
                                          "func": "config_terms",
                                          "sendevent": True,
                                          "noexpand": True}], ])

        return panels

    def get_panel1_value(self):
        values = super(EM3D_ColdPlasma, self).get_panel1_value()
        values.extend([self.col_model, self.stix_terms_str(), self])
        return values

    def import_panel1_value(self, v):
        check = super(EM3D_ColdPlasma, self).import_panel1_value(v[:-3])
        self.col_model = v[-3]
        return check

    @property
    def jited_coeff(self):
        return self._jited_coeff

    def compile_coeffs(self):
        self._jited_coeff = self.get_coeffs()

    def get_coeffs(self):
        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        B, dens_e, t_e, dens_i, masses, charges = self.vt.make_value_or_expression(
            self)
        ind_vars = self.get_root_phys().ind_vars
        col_model = self.col_model

        from petram.phys.common.rf_dispersion_coldplasma import build_coefficients
        coeff1, coeff2, coeff3, coeff5 = build_coefficients(ind_vars, omega, B, dens_e, t_e,
                                                                    dens_i, masses, charges, col_model, cnorm,
                                                                    self._global_ns, self._local_ns,
                                                                    sdim=3, terms=self.stix_terms)

        return coeff1, coeff2, coeff3, coeff5

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

        coeff1, coeff2, coeff3, _coeff_nuei = self.jited_coeff
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
        from petram.helper.variables import add_constant

        if len(self._sel_index) == 0:
            return

        freq, omega = self.get_root_phys().get_freq_omega()
        B, dens_e, t_e, dens_i, masses, charges = self.vt.make_value_or_expression(
            self)
        col_model = self.col_model

        from petram.phys.common.rf_dispersion_coldplasma import build_variables

        ss = self.parent.parent.name()+'_'+self.name()  # phys module name + name
        ret = build_variables(v, ss, ind_vars, omega, B, dens_e, t_e,
                              dens_i, masses, charges, col_model,
                              self._global_ns, self._local_ns,
                              terms=self.stix_terms,)

        from petram.phys.common.rf_dispersion_coldplasma import add_domain_variables_common
        add_domain_variables_common(self, ret, v, suffix, ind_vars)

        var = ['x', 'y', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        return
