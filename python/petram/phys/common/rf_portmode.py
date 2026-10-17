'''
  Definition of 3D port modes

  TE Mode:
    Expression are based on Microwave Engineering p122 - p123.
    Note that it consists from two terms
       1)  \int dS W \dot n \times iwH (VectorFETangentIntegrator does this)
       2)  an weighting to evaulate mode amplutude from the E field
           on a boundary
  TEM Mode:
       E is parallel to the periodic edge
       Mode number is ignored (of course)

  CoaxTEM:

  CircularTE:

  About sign of port phasing.
       positive phasing means the incoming wave appears later (phase delay)
       at the port

  Note: jcH returns H/mu0 (which is B)
'''
from petram.mfem_config import use_parallel
from numpy import sin, cos, exp, sqrt, array, sum, pi, log, cross, arctan2

from petram.phys.phys_const import epsilon0, mu0

import petram.debug as debug
dprint1, dprint2, dprint3 = debug.init_dprints('EM3D_PortMode')

if use_parallel:
    import mfem.par as mfem
    from mfem.common.mpi_debug import nicePrint
    '''
    from mpi4py import MPI
    num_proc = MPI.COMM_WORLD.size
    myid     = MPI.COMM_WORLD.rank
    '''
else:
    import mfem.ser as mfem
    nicePrint = dprint1

'''
   rectangular wg TE
'''


def get_portmode_coeff_cls(mode):
    if mode == 'TEM':
        return C_Et_TEM, C_jwHt_TEM
    elif mode == 'TE':
        return C_Et_TE, C_jwHt_TE
    elif mode == 'Coax(TEM)':
        return C_Et_CoaxTEM, C_jwHt_CoaxTEM
    elif mode == 'Circular(TE)':
        return C_Et_CircularTE, C_jwHt_CircularTE
    else:
        raise NotImplementedError(
            "you must implement this mode")


def TE_norm(m, n, a, b, alpha, gamma):
    if m == 0:
        return sqrt(a*b * alpha*gamma/8*n*n/b/b*2)
    elif n == 0:
        return sqrt(a*b * alpha*gamma/8*m*m/a/a*2)
    else:
        return sqrt(a*b * alpha*gamma/8*(m*m/a/a + n*n/b/b))


class C_Et_TE(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0,
                 m=0, n=1):
        self.real = real
        self.a, self.b, self.c = bdry.a, bdry.b, bdry.c
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.m, self.n = m, n
        freq, omega = bdry.get_root_phys().get_freq_omega()

        k = eps*omega*sqrt(eps*epsilon0 * mur*mu0)
        kc = sqrt((self.m*pi/bdry.a)**2 +
                  (self.n*pi/bdry.b)**2)
        beta = sqrt(k**2 - kc**2)
        # self.AA = 1.0 ##
        alpha = omega*mur*mu0*pi/kc/kc
        gamma = beta*pi/kc/kc
        norm = TE_norm(self.m, self.n, self.a, self.b, alpha, gamma)
        self.AA = alpha/norm*amp * exp(1j*phase*pi/180.)
        dprint2("normalization to old ", self.AA)
        mfem.VectorPyCoefficient.__init__(self, sdim)

    def EvalValue(self, x):
        p = array(x)
        # x, y is positive (first quadrant)
        x = abs(sum((p - self.c)*self.a_vec))
        y = abs(sum((p - self.c)*self.b_vec))

        Ex = self.AA * self.n/self.b*(
            cos(self.m*pi*x/self.a) *
            sin(self.n*pi*y/self.b))
        Ey = - self.AA * self.m/self.a*(
            sin(self.m*pi*x/self.a) *
            cos(self.n*pi*y/self.b))

        E = Ex*self.a_vec + Ey*self.b_vec
        if self.real:
            return -E.real
        else:
            return -E.imag


class C_jwHt_TE(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0,
                 cnorm=1.0, m=0, n=1):
        self.real = real
        self.phase = phase  # phase !=0 for incoming wave

        freq, omega = bdry.get_root_phys().get_freq_omega()
        # cnorm = bdry.get_root_phys().get_coeff_norm()

        self.a, self.b, self.c = bdry.a, bdry.b, bdry.c
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.m, self.n = m, n

        k = eps*omega*sqrt(eps*epsilon0 * mur*mu0)
        kc = sqrt((self.m*pi/bdry.a)**2 +
                  (self.n*pi/bdry.b)**2)
        if kc > k:
            raise ValueError('Mode does not propagate')
        beta = sqrt(k**2 - kc**2)
        dprint1("propagation constant:" + str(beta))

        alpha = omega*mur*mu0*pi/kc/kc
        gamma = beta*pi/kc/kc
        norm = TE_norm(self.m, self.n, self.a, self.b, alpha, gamma)
        self.AA = omega*gamma/norm*amp*cnorm

        # AA = omega*mur*mu0*pi/kc/kc*amp
        # self.AA = omega*beta*pi/kc/kc/AA
        # self.AA = omega*beta*pi/kc/kc*amp/1000.

        mfem.VectorPyCoefficient.__init__(self, sdim)

    def EvalValue(self, x):
        p = array(x)
        # x, y is positive (first quadrant)
        x = abs(sum((p - self.c)*self.a_vec))
        y = abs(sum((p - self.c)*self.b_vec))

        Hx = 1j*self.AA * self.m/self.a*(
            sin(self.m*pi*x/self.a) *
            cos(self.n*pi*y/self.b))
        Hy = 1j*self.AA * self.n/self.b*(
            cos(self.m*pi*x/self.a) *
            sin(self.n*pi*y/self.b))

        H = Hx*self.a_vec + Hy*self.b_vec

        H = H * exp(1j*self.phase/180.*pi)
        if self.real:
            return H.real
        else:
            return H.imag


'''
   rectangular port parallel metal TEM
'''


class C_Et_TEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0, **kargs):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()
        self.real = real
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.AA = 1.0*amp * exp(1j*phase*pi/180.)

    def EvalValue(self, x):
        Ex = self.AA
        E = -Ex*self.b_vec
        if self.real:
            return -E.real
        else:
            return -E.imag


class C_jwHt_TEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0, cnorm=1.0, **kargs):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()
        # cnorm = bdry.get_root_phys().get_coeff_norm()

        self.real = real
        self.phase = phase  # phase !=0 for incoming wave
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.AA = omega*sqrt(epsilon0*eps/mu0/mur)*amp*cnorm

    def EvalValue(self, x):
        Hy = 1j*self.AA
        H = Hy*self.a_vec
        H = H * exp(1j*self.phase*pi/180.)
        if self.real:
            return H.real
        else:
            return H.imag


'''
   coax port TEM
'''


def coax_norm(a, b, mur, eps):
    return sqrt(1/pi/sqrt(epsilon0*eps/mu0/mur)/log(b/a))


class C_Et_CoaxTEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0, **kargs):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.real = real
        self.a = bdry.a
        self.b = bdry.b
        self.ctr = bdry.ctr
        self.AA = coax_norm(self.a, self.b, mur, eps) * \
            amp * exp(1j*phase*pi/180.)

    def EvalValue(self, x):
        r = (x - self.ctr)
        rho = sqrt(sum(r**2))
        nr = r/rho

#       E = nr*log(self.b/rho)/log(self.b/self.a)
#       E = nr/log(self.b/self.a)/rho
        E = nr/rho*self.AA
        if self.real:
            return E.real
        else:
            return E.imag


class C_jwHt_CoaxTEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0, cnorm=1.0, **kargs):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()
        # cnorm = bdry.get_root_phys().get_coeff_norm()

        self.real = real
        self.norm = bdry.norm
        self.a = bdry.a
        self.b = bdry.b
        self.ctr = bdry.ctr
        self.phase = phase  # phase !=0 for incoming wave
        # self.AA = omega*sqrt(epsilon0*eps/mu0/mur)
        self.AA = coax_norm(self.a, self.b, mur, eps) * \
            omega*sqrt(epsilon0*eps/mu0/mur)*amp*cnorm

    def EvalValue(self, x):
        r = (x - self.ctr)
        rho = sqrt(sum(r**2))
        nr = r/rho
        nr = cross(nr, self.norm)
#       H = nr*log(self.b/rho)/log(self.b/self.a)
#       H = nr/log(self.b/self.a)/rho
        H = 1j*nr/rho*self.AA
        # H = 1j*self.AA*H
        H = H * exp(1j*self.phase*pi/180.)
        if self.real:
            return -H.real
        else:
            return -H.imag


'''
   circular port TEM
'''


def circular_norm():
    # ToDo
    return 1.0


class C_CircularTE(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0, amp=1.0, phase=0.0, cnorm=1.0,
                 m=0, n=1):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.real = real
        self.m = m
        self.n = n
        self.a = bdry.a  #
        self.ctr = bdry.ctr
        self.ax1 = bdry.ax1
        self.ax2 = bdry.ax2
        self.norm = bdry.norm
        self.cnorm = cnorm
        self.phase = phase

        self.amp = amp

        freq, omega = bdry.get_root_phys().get_freq_omega()
        self.omega = omega

        from scipy.special import jnp_zeros

        xzero = jnp_zeros(abs(self.m), self.n)[-1]
        kc = xzero/self.a
        k = eps*omega*sqrt(epsilon0*eps * mu0*mur)
        if k**2 < kc**2:
            assert False, "mode does not propagte"

        self.kg = sqrt(k**2-kc**2)
        self.kc = kc
        #print("kc", self.kc)
        self.AA = circular_norm() * amp * exp(1j*phase*pi/180.)

    def compute_jv_x(self, rr):
        from scipy.special import jv
        from math import factorial

        m = abs(self.m)
        kc = self.kc

        if m > 0:
            x = rr*kc
            if x < 0.001:
                jv_x = 0
                for i in range(5):
                  jv_x += (-1)**i /factorial(i)/factorial(m+i) *x**(2*i + m-1)/2**(2*i + m)
            else:
                jv_x = jv(m, x)/x
            jv_x = jv_x*kc
        else:
            return 0

        if m < 0:
            jv_x*(-1)**m
        return jv_x

class C_Et_CircularTE(C_CircularTE):
    def EvalValue(self, x):
        r = (x - self.ctr)
        rr = sqrt(sum(r**2))
        th = arctan2(sum(r*self.ax2), sum(r*self.ax1))

        nr = self.ax1*cos(th) + self.ax2*sin(th)
        nt = -self.ax1*sin(th) + self.ax2*cos(th)

        jv_x = self.compute_jv_x(rr)

        from scipy.special import jvp
        
        if self.m > 0:
            Er = self.m * jv_x*exp(1j*self.m*th-1j*pi/2)
        else:
            Er = self.m * jv_x*exp(1j*self.m*th-1j*pi/2)*(-1)**self.m

        Et = self.kc*jvp(self.m, rr*self.kc)*exp(1j*self.m*th)


        E = (Er*nr + Et*nt) * self.AA

        if self.real:
            return E.real
        else:
            return E.imag


class C_jwHt_CircularTE(C_CircularTE):
    def EvalValue(self, x):
        r = (x - self.ctr)
        rr = sqrt(sum(r**2))
        th = arctan2(sum(r*self.ax2), sum(r*self.ax1))

        nr = self.ax1*cos(th) + self.ax2*sin(th)
        nt = -self.ax1*sin(th) + self.ax2*cos(th)

        jv_x = self.compute_jv_x(rr)

        from scipy.special import jvp

        # Hr = -self.kg*self.kc/self.omega*jvp(self.m, rr*self.kc)*cos(self.m*th)
        # Ht =  self.kg*self.m/self.omega/rr*jv(self.m, rr*self.kc)*sin(self.m*th)
        Hr = self.kg*self.kc/self.omega * \
            jvp(self.m, rr*self.kc)*exp(1j*self.m*th)
        #Ht = self.kg*self.m/self.omega/rr * \
        #    jv(self.m, rr*self.kc)*exp(1j*self.m*th)
        if self.m >= 0:
            Ht = -self.kg*self.m/self.omega*jv_x*exp(1j*self.m*th-1j*pi/2)
        else:
            Ht = -self.kg*self.m/self.omega*jv_x*exp(1j*self.m*th-1j*pi/2)*(-1)**self.m

        H = 1j*self.omega*(Hr*nr + Ht*nt) * self.AA*self.cnorm /mu0

        if self.real:
            return -H.real
        else:
            return -H.imag
