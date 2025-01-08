'''
  Definition of 3D port modes

  TE Mode
    Expression are based on Microwave Engineering p122 - p123.
    Note that it consists from two terms
       1)  \int dS W \dot n \times iwH (VectorFETangentIntegrator does this)
       2)  an weighting to evaulate mode amplutude from the E field
           on a boundary              
  TEM Mode
       E is parallel to the periodic edge
       Mode number is ignored (of course)

  About sign of port phasing.
       positive phasing means the incoming wave appears later (phase delay)
       at the port
'''
from petram.mfem_config import use_parallel
import numpy as np

from petram.phys.em3d.em3d_const import epsilon0, mu0

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


def TE_norm(m, n, a, b, alpha, gamma):
    if m == 0:
        return np.sqrt(a*b * alpha*gamma/8*n*n/b/b*2)
    elif n == 0:
        return np.sqrt(a*b * alpha*gamma/8*m*m/a/a*2)
    else:
        return np.sqrt(a*b * alpha*gamma/8*(m*m/a/a + n*n/b/b))


class C_Et_TE(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0):
        self.real = real
        self.a, self.b, self.c = bdry.a, bdry.b, bdry.c
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.m, self.n = bdry.mn[0], bdry.mn[1]
        freq, omega = bdry.get_root_phys().get_freq_omega()

        k = omega*np.sqrt(eps*epsilon0 * mur*mu0)
        kc = np.sqrt((bdry.mn[0]*np.pi/bdry.a)**2 +
                     (bdry.mn[1]*np.pi/bdry.b)**2)
        beta = np.sqrt(k**2 - kc**2)
        #self.AA = 1.0 ##
        alpha = omega*mur*mu0*np.pi/kc/kc
        gamma = beta*np.pi/kc/kc
        norm = TE_norm(self.m, self.n, self.a, self.b, alpha, gamma)
        self.AA = alpha/norm
        dprint2("normalization to old ", self.AA)
        mfem.VectorPyCoefficient.__init__(self, sdim)

    def EvalValue(self, x):
        p = np.array(x)
        # x, y is positive (first quadrant)
        x = abs(np.sum((p - self.c)*self.a_vec))
        y = abs(np.sum((p - self.c)*self.b_vec))

        Ex = self.AA * self.n/self.b*(
            np.cos(self.m*np.pi*x/self.a) *
            np.sin(self.n*np.pi*y/self.b))
        Ey = - self.AA * self.m/self.a*(
            np.sin(self.m*np.pi*x/self.a) *
            np.cos(self.n*np.pi*y/self.b))

        E = Ex*self.a_vec + Ey*self.b_vec
        if self.real:
            return -E.real
        else:
            return -E.imag


class C_jwHt_TE(mfem.VectorPyCoefficient):
    def __init__(self, sdim, phase, bdry, real=True, amp=1.0, eps=1.0, mur=1.0):
        self.real = real
        self.phase = phase  # phase !=0 for incoming wave

        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.a, self.b, self.c = bdry.a, bdry.b, bdry.c
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.m, self.n = bdry.mn[0], bdry.mn[1]

        k = omega*np.sqrt(eps*epsilon0 * mur*mu0)
        kc = np.sqrt((bdry.mn[0]*np.pi/bdry.a)**2 +
                     (bdry.mn[1]*np.pi/bdry.b)**2)
        if kc > k:
            raise ValueError('Mode does not propagate')
        beta = np.sqrt(k**2 - kc**2)
        dprint1("propagation constant:" + str(beta))

        alpha = omega*mur*mu0*np.pi/kc/kc
        gamma = beta*np.pi/kc/kc
        norm = TE_norm(self.m, self.n, self.a, self.b, alpha, gamma)
        self.AA = omega*gamma/norm*amp

        #AA = omega*mur*mu0*np.pi/kc/kc*amp
        #self.AA = omega*beta*np.pi/kc/kc/AA
        #self.AA = omega*beta*np.pi/kc/kc*amp/1000.

        mfem.VectorPyCoefficient.__init__(self, sdim)

    def EvalValue(self, x):
        p = np.array(x)
        # x, y is positive (first quadrant)
        x = abs(np.sum((p - self.c)*self.a_vec))
        y = abs(np.sum((p - self.c)*self.b_vec))

        Hx = 1j*self.AA * self.m/self.a*(
            np.sin(self.m*np.pi*x/self.a) *
            np.cos(self.n*np.pi*y/self.b))
        Hy = 1j*self.AA * self.n/self.b*(
            np.cos(self.m*np.pi*x/self.a) *
            np.sin(self.n*np.pi*y/self.b))

        H = Hx*self.a_vec + Hy*self.b_vec

        H = H * np.exp(1j*self.phase/180.*np.pi)
        if self.real:
            return H.real
        else:
            return H.imag


'''
   rectangular port parallel metal TEM
'''


class C_Et_TEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()
        self.real = real
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.AA = 1.0

    def EvalValue(self, x):
        Ex = self.AA
        E = -Ex*self.b_vec
        if self.real:
            return -E.real
        else:
            return -E.imag


class C_jwHt_TEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, phase, bdry, real=True, amp=1.0, eps=1.0, mur=1.0):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.real = real
        self.phase = phase  # phase !=0 for incoming wave
        self.a_vec, self.b_vec = bdry.a_vec, bdry.b_vec
        self.AA = omega*np.sqrt(epsilon0*eps/mu0/mur)*amp

    def EvalValue(self, x):
        Hy = 1j*self.AA
        H = Hy*self.a_vec
        H = H * np.exp(1j*self.phase*np.pi/180.)
        if self.real:
            return H.real
        else:
            return H.imag


'''
   coax port TEM
'''


def coax_norm(a, b, mur, eps):
    return np.sqrt(1/np.pi/np.sqrt(epsilon0*eps/mu0/mur)/np.log(b/a))


class C_Et_CoaxTEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, bdry, real=True, eps=1.0, mur=1.0):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.real = real
        self.a = bdry.a
        self.b = bdry.b
        self.ctr = bdry.ctr
        self.AA = coax_norm(self.a, self.b, mur, eps)

    def EvalValue(self, x):
        r = (x - self.ctr)
        rho = np.sqrt(np.sum(r**2))
        nr = r/rho

#       E = nr*np.log(self.b/rho)/np.log(self.b/self.a)
#       E = nr/np.log(self.b/self.a)/rho
        E = nr/rho*self.AA
        if self.real:
            return E.real
        else:
            return E.imag


class C_jwHt_CoaxTEM(mfem.VectorPyCoefficient):
    def __init__(self, sdim, phase, bdry, real=True, amp=1.0, eps=1.0, mur=1.0):
        mfem.VectorPyCoefficient.__init__(self, sdim)
        freq, omega = bdry.get_root_phys().get_freq_omega()

        self.real = real
        self.norm = bdry.norm
        self.a = bdry.a
        self.b = bdry.b
        self.ctr = bdry.ctr
        self.phase = phase  # phase !=0 for incoming wave
        #self.AA = omega*np.sqrt(epsilon0*eps/mu0/mur)
        self.AA = coax_norm(self.a, self.b, mur, eps) * \
            omega*np.sqrt(epsilon0*eps/mu0/mur)*amp

    def EvalValue(self, x):
        r = (x - self.ctr)
        rho = np.sqrt(np.sum(r**2))
        nr = r/rho
        nr = np.cross(nr, self.norm)
#       H = nr*np.log(self.b/rho)/np.log(self.b/self.a)
#       H = nr/np.log(self.b/self.a)/rho
        H = 1j*nr/rho*self.AA
        #H = 1j*self.AA*H
        H = H * np.exp(1j*self.phase*np.pi/180.)
        if self.real:
            return -H.real
        else:
            return -H.imag
