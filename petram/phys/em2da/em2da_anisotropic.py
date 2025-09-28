'''
   Vacuum region:
      However, can have arbitrary scalar epsilon_r, mu_r, sigma


'''
import numpy as np

from petram.phys.phys_const import mu0, epsilon0

from petram.phys.phys_model  import PhysCoefficient, VectorPhysCoefficient
from petram.phys.phys_model  import MatrixPhysCoefficient, Coefficient_Evaluator
from petram.phys.em2da.em2da_base import EM2Da_Bdry, EM2Da_Domain

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM2Da_Anisotropic')

from petram.mfem_config import use_parallel
if use_parallel:
   import mfem.par as mfem
else:
   import mfem.ser as mfem

from petram.phys.vtable import VtableElement, Vtable
data =  (('epsilonr', VtableElement('epsilonr', type='complex',
                                     guilabel = 'epsilonr',
                                     suffix =[('r', 'phi', 'z'), ('r', 'phi', 'z')],
                                     default = np.eye(3, 3),
                                     tip = "relative permittivity" )),
         ('mur', VtableElement('mur', type='complex',
                                     guilabel = 'mur',
                                     default = 1.0,
                                     tip = "relative permeability" )),
         ('sigma', VtableElement('sigma', type='complex',
                                     guilabel = 'sigma',
                                     suffix =[('r', 'phi', 'z'), ('r', 'phi', 'z')],
                                     default = np.zeros((3, 3)),
                                     tip = "contuctivity" )),
         ('t_mode', VtableElement('t_mode', type='float',
                                     guilabel = 'm',
                                     default = 0.0,
                                     tip = "mode number" )),)

'''
Expansion of matrix is as follows

               [e_rz  e_12 ][Erz ]
[Wrz, Wphx] =  [           ][    ] = Wrz e_rz Erz + Wrz e_12 Ephi
               [e_21  e_phi][Ephx]

                                + Wphi e_21 Erz + Wphi*e_phi*Ephi


  Erz = Er e_r + Ez e_z
  Ephx = rho Ephi
'''
class M_RZ(MatrixPhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.omega = kwargs.pop('omega', 1.0)
        self.cnorm = kwargs.pop('cnorm')
        super(M_RZ, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = Coefficient_Evaluator.EvalValue(self, x).reshape(3,3)
        val = self.apply_factor(x, val)
        return val[[0,2], :][:, [0, 2]]

class M_12(VectorPhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.omega = kwargs.pop('omega', 1.0)
        self.cnorm = kwargs.pop('cnorm')
        super(M_12, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = Coefficient_Evaluator.EvalValue(self, x).reshape(3,3)
        val = self.apply_factor(x, val)
        return val[[0, 2], [1]]

class M_21(VectorPhysCoefficient):
    def __init__(self, *args, **kwargs):
        self.omega = kwargs.pop('omega', 1.0)
        self.cnorm = kwargs.pop('cnorm')
        super(M_21, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = Coefficient_Evaluator.EvalValue(self, x).reshape(3,3)
        val = self.apply_factor(x, val)
        return val[[1], [0,2]]

class M_PHI(PhysCoefficient):
    def __init__(self, *args, **kwargs):
        kwargs['isArray'] = True
        self.omega = kwargs.pop('omega', 1.0)
        self.cnorm = kwargs.pop('cnorm')
        super(M_PHI, self).__init__(*args, **kwargs)
    def EvalValue(self, x):
        val = Coefficient_Evaluator.EvalValue(self, x).reshape(3,3)
        val = self.apply_factor(x, val)
        return val[1, 1]

class eps_o_r(object):
    def apply_factor(self, x, v):
        v = - v * epsilon0 * self.omega * self.omega / x[0] * self.cnorm
        if self.real:  return v.real
        else: return v.imag
class eps_x_r(object):
    def apply_factor(self, x, v):
        v = - v * epsilon0 * self.omega * self.omega * x[0]* self.cnorm
        if self.real:  return v.real
        else: return v.imag
class eps(object):
    def apply_factor(self, x, v):
        v = - v * epsilon0 * self.omega * self.omega * self.cnorm
        if self.real:  return v.real
        else: return v.imag

class sigma_o_r(object):
    def apply_factor(self, x, v):
        v = - 1j * self.omega * v / x[0] * self.cnorm
        if self.real:  return v.real
        else: return v.imag
class sigma_x_r(object):
    def apply_factor(self, x, v):
        v = - 1j * self.omega * v * x[0] * self.cnorm
        if self.real:  return v.real
        else: return v.imag
class sigma(object):
    def apply_factor(self, x, v):
        v = - 1j * self.omega * v * self.cnorm
        if self.real:  return v.real
        else: return v.imag

#class Epsilon_o_r_rz(M_RZ, eps_o_r):
#    pass
class Epsilon_o_r_phi(M_PHI, eps_o_r):
    pass
#class Sigma_o_r_rz(M_RZ, sigma_o_r):
#    pass
class Sigma_o_r_phi(M_PHI, sigma_o_r):
    pass

class Epsilon_x_r_rz(M_RZ, eps_x_r):
    pass
class Epsilon_x_r_phi(M_PHI, eps_x_r):
    pass
class Sigma_x_r_rz(M_RZ, sigma_x_r):
    pass
class Sigma_x_r_phi(M_PHI, sigma_x_r):
    pass


class Epsilon_21(M_21, eps):
    pass

class Epsilon_12(M_12, eps):
    pass

class Sigma_12(M_12, sigma):
    pass

class Sigma_21(M_21, sigma):
    pass

class InvMu_x_r(PhysCoefficient):
   '''
      r/mu0/mur
   '''
   def __init__(self, *args, **kwargs):
       self.cnorm = kwargs.pop('cnorm')
       super(InvMu_x_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(InvMu_x_r, self).EvalValue(x)
       v = 1/mu0/v*x[0] * self.cnorm
       if self.real:  return v.real
       else: return v.imag

class InvMu_o_r(PhysCoefficient):
   '''
      1/mu0/mur/r
   '''
   def __init__(self, *args, **kwargs):
       self.tmode = kwargs.pop('tmode', 0.0)
       self.cnorm = kwargs.pop('cnorm')
       super(InvMu_o_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(InvMu_o_r, self).EvalValue(x)
       v = 1./mu0/v/x[0] * self.cnorm
       if self.real:  return v.real
       else: return v.imag

class iInvMu_m_o_r(PhysCoefficient):
   '''
      1j/mu0/mur/r
   '''
   def __init__(self, *args, **kwargs):
       self.tmode = kwargs.pop('tmode', 1.0)
       self.cnorm = kwargs.pop('cnorm')
       super(iInvMu_m_o_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(iInvMu_m_o_r, self).EvalValue(x)
       v = 1j/mu0/v/x[0]*self.tmode * self.cnorm
       if self.real:  return v.real
       else: return v.imag

class InvMu_m2_o_r(PhysCoefficient):
   '''
      1./mu0/mur/r/m^2
   '''
   def __init__(self, *args, **kwargs):
       self.tmode = kwargs.pop('tmode', 1.0)
       self.cnorm = kwargs.pop('cnorm')
       super(InvMu_m2_o_r, self).__init__(*args, **kwargs)

   def EvalValue(self, x):
       v = super(InvMu_m2_o_r, self).EvalValue(x)
       v = 1/mu0/v/x[0]*self.tmode*self.tmode *self.cnorm
       if self.real:  return v.real
       else: return v.imag

def domain_constraints():
   return [EM2Da_Anisotropic]

class EM2Da_Anisotropic(EM2Da_Domain):
    vt  = Vtable(data)
    #nlterms = ['epsilonr']

    def has_bf_contribution(self, kfes):
        if kfes == 0: return True
        elif kfes == 1: return True
        else: return False

    def has_mixed_contribution(self):
        return True

    def get_mixedbf_loc(self):
        '''
        r, c, and flag1, flag2 of MixedBilinearForm
           flag1 : take transpose
           flag2 : take conj
        '''
        return [(0, 1, 1, 1), (1, 0, 1, 1),]#(0, 1, -1, 1)]

    def add_bf_contribution(self, engine, a, real = True, kfes=0):
        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        e, m, s, tmode = self.vt.make_value_or_expression(self)
        if not isinstance(e, str): e = str(e)
        if not isinstance(m, str): m = str(m)
        if not isinstance(s, str): s = str(s)

        if kfes == 0: ## ND element (Epoloidal)
            if real:
                dprint1("Add ND contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add ND contribution(imag)" + str(self._sel_index))
            imu_x_r = InvMu_x_r(m,  self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, cnorm=cnorm)
            s_x_r = Sigma_x_r_rz(2, s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega, cnorm=cnorm)
            e_x_r = Epsilon_x_r_rz(2, e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega, cnorm=cnorm)

            self.add_integrator(engine, 'mur', imu_x_r,
                                a.AddDomainIntegrator,
                                mfem.CurlCurlIntegrator)
            self.add_integrator(engine, 'epsilonr', e_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)
            self.add_integrator(engine, 'sigma', s_x_r,
                                a.AddDomainIntegrator,
                                mfem.VectorFEMassIntegrator)

            if tmode != 0:
                imu_o_r_2 = InvMu_m2_o_r(m,  self.get_root_phys().ind_vars,
                                      self._local_ns, self._global_ns,
                                      real = real, tmode = tmode, cnorm=cnorm)
                self.add_integrator(engine, 'mur', imu_o_r_2,
                                    a.AddDomainIntegrator,
                                    mfem.VectorFEMassIntegrator)

        elif kfes == 1: ## H1 element (Etoroidal)
            if real:
                dprint1("Add H1 contribution(real)" + str(self._sel_index))
            else:
                dprint1("Add H1 contribution(imag)" + str(self._sel_index))
            imv_o_r_1 = InvMu_o_r(m,  self.get_root_phys().ind_vars,
                            self._local_ns, self._global_ns,
                            real = real)
            e_o_r = Epsilon_o_r_phi(e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega)
            s_o_r = Sigma_o_r_phi(s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega)

            self.add_integrator(engine, 'mur', imv_o_r_1,
                                a.AddDomainIntegrator,
                                mfem.DiffusionIntegrator)
            self.add_integrator(engine, 'epsilonr', e_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
            self.add_integrator(engine, 'sigma', s_o_r,
                                a.AddDomainIntegrator,
                                mfem.MassIntegrator)
        else:
            pass

    def add_mix_contribution(self, engine, mbf, r, c, is_trans, real = True):
        if real:
            dprint1("Add mixed contribution(real)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))
        else:
            dprint1("Add mixed contribution(imag)" + "(" + str(r) + "," + str(c) +')'
                    +str(self._sel_index))

        freq, omega = self.get_root_phys().get_freq_omega()
        cnorm = self.get_root_phys().get_coeff_norm()

        e, m, s, tmode = self.vt.make_value_or_expression(self)

        if not isinstance(e, str): e = str(e)
        if not isinstance(m, str): m = str(m)
        if not isinstance(s, str): s = str(s)

        imv_o_r_3 = iInvMu_m_o_r(m,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                                 real = real, tmode = tmode, cnorm=cnorm)
        if r == 1 and c == 0:
            e = Epsilon_21(2, e, self.get_root_phys().ind_vars,
                                self._local_ns, self._global_ns,
                                real = real, omega = omega, cnorm=cnorm)
            s = Sigma_21(2, s,  self.get_root_phys().ind_vars,
                              self._local_ns, self._global_ns,
                              real = real, omega = omega, cnorm=cnorm)
            # (a_vec dot u_vec, v_scalar)
            itg = mfem.MixedDotProductIntegrator
            self.add_integrator(engine, 'epsilon', e,
                                mbf.AddDomainIntegrator, itg)
            self.add_integrator(engine, 'sigma', s,
                                mbf.AddDomainIntegrator, itg)
            # (-a u_vec, div v_scalar)
            itg =  mfem.MixedVectorWeakDivergenceIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                                mbf.AddDomainIntegrator, itg)

        else:
            e = Epsilon_12(2, e, self.get_root_phys().ind_vars,
                             self._local_ns, self._global_ns,
                             real = real, omega = omega, cnorm=cnorm)
            s = Sigma_12(2, s,  self.get_root_phys().ind_vars,
                           self._local_ns, self._global_ns,
                           real = real, omega = omega, cnorm=cnorm)

             #itg = mfem.MixedDotProductIntegrator
            itg = mfem.MixedVectorProductIntegrator
            self.add_integrator(engine, 'epsilon', e,
                             mbf.AddDomainIntegrator, itg)
            self.add_integrator(engine, 'sigma', s,
                             mbf.AddDomainIntegrator, itg)
            # (a grad u_scalar, v_vec)

            itg =  mfem.MixedVectorGradientIntegrator
            self.add_integrator(engine, 'mur', imv_o_r_3,
                             mbf.AddDomainIntegrator, itg)

    def add_domain_variables(self, v, n, suffix, ind_vars):
        from petram.helper.variables import add_constant

        e, m, s, tmode = self.vt.make_value_or_expression(self)

        self.do_add_matrix_expr(v, suffix, ind_vars, 'epsilonr', e)
        self.do_add_scalar_expr(v, suffix, ind_vars, 'smur', m, add_diag=3)
        self.do_add_matrix_expr(v, suffix, ind_vars, 'sigma', s)

        var = ['r', 'phi', 'z']
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'epsilonr')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'mur')
        self.do_add_matrix_component_expr(v, suffix, ind_vars, var, 'sigma')

        add_constant(v, 'm_mode', suffix, np.float64(tmode),
                     domains = self._sel_index,
                     gdomain = self._global_ns)
