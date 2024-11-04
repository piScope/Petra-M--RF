'''
 non-relativistic Maxwellian hot plasma dispersion, icluding
    all order in k_perp

 max nherm is 20

  epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne, npara, nperp, nhrms)

      temps : ion temperature (eV)
      masses : ion masses (kg)
      charges : ion charges (C/1.6e-19)

 (sample run)
    >>> nperp,npara,denses,masses,charges,temps,ne,Te,Bmang,w,nhrms=(40.,10.,[5.e19,5.e18],[2*Da,Da],[1, 1],[15e3,15e3],5.e19,10e3,0.5,3e7*2*3.1415926,20)
    >>> from petram.phys.common.rf_dispersion_lkplasma_numba import epsilonr_pl_hot_std
    >>> epsilonr_pl_hot_std(w,np.array(B),np.array(temps),np.array(denses),np.array(masses),np.array(charges),Te,ne,npara,nperp,nhrms)


array([[-1.56316016e+03+1.23807373e+01j, -1.12033076e+01-9.84907588e+03j,
        -1.22779569e+01-8.12259897e-01j],
       [ 1.12033076e+01+9.84907588e+03j, -2.06928780e+03+9.04122672e+02j,
         2.66395405e+04-2.17944393e+04j],
       [-1.22779569e+01-8.12259897e-01j, -2.66395405e+04+2.17944393e+04j,
         1.29769916e+06+1.58788920e+06j]])

'''
from petram.phys.phys_const import c as speed_of_light
from petram.helper.numba_ive import ive
from petram.phys.common.numba_zfunc import zfunc
from numpy import pi, sqrt
from petram.phys.phys_const import c_cgs as clight
from petram.phys.phys_const import (mass_electron, mass_proton)
from petram.phys.phys_const import q0_cgs
from petram.phys.phys_const import epsilon0 as e0
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import Da
from numba import njit, void, int32, int64, float64, complex128, types
from numpy import (pi, sin, cos, exp, sqrt, log, arctan2, cross,
                   max, array, linspace, conj, transpose,
                   sum, zeros, dot, array, ascontiguousarray)
import numpy as np

iarray_ro = types.Array(int32, 1, 'C', readonly=True)
darray_ro = types.Array(float64, 1, 'C', readonly=True)


# constant

# constants
gausspertesla = 1.E4
meter3_per_cm3 = 1.0E-6
me_gram = mass_electron * 1000.
mp_gram = mass_proton * 1000.
ergperkev = q_base * 1e10
qe = -q_base


@njit("float64(float64)")
def om(freq):
    return 2.0 * pi * freq


@njit("float64(float64, float64)")
def wpesq(ne, freq):
    return 4.0 * pi * ne * meter3_per_cm3 * q0_cgs**2 / me_gram / om(freq)**2


@njit("float64(float64, float64, float64, float64)")
def wpisq(ni, A, Z, freq):
    return 4.0 * pi * ni * meter3_per_cm3 * \
        (q0_cgs * Z)**2 / (A * mp_gram) / om(freq)**2


@njit("float64(float64, float64)")
def wce(Bmagn, freq):
    return (-q0_cgs * Bmagn * gausspertesla) / (me_gram * clight) / om(freq)


@njit("float64(float64, float64, float64, float64)")
def wci(Bmagn, freq, A, Z):
    return (q0_cgs * Z * Bmagn * gausspertesla) / (A * mp_gram * clight) / om(freq)


@njit("float64(float64)")
def vte(te_kev):
    tee = ergperkev * te_kev
    return sqrt(2.0 * tee / me_gram) / clight


@njit("float64(float64, float64)")
def vti(ti_kev, A):
    tii = ergperkev * ti_kev
    return sqrt(2.0 * tii / (A * mp_gram)) / clight


@njit("float64(float64, float64, float64, float64, float64, float64)")
def lam_i(nperp, ti_kev, Bmagn, freq, A, Z):
    """Lambda for ions in Stix notation (Eq. (10-55) Stix's book, p. 258 ).
    """
    return nperp**2 * vti(ti_kev, A)**2 / (2.0 * wci(Bmagn, freq, A, Z)**2)


@njit("float64(float64, float64, float64, float64)")
def lam_e(nperp, te_kev, Bmagn, freq):
    """Lambda for electrons in Stix notation (Eq. (10-55) Stix's book, p. 258 ).
    """
    return nperp**2 * vte(te_kev)**2 / (2.0 * wce(Bmagn, freq)**2)


@njit(void(complex128[:, :], int64, int64))
def print_mat(mat, r, c):
    '''
    routine for debugging
    '''
    for i in range(r):
        for j in range(c):
            print("mat r="+str(i) + ", z="+str(j) + " :")
            print(mat[i, j])


@njit("float64(float64, float64, float64, float64, float64, float64, float64)")
def IminusIp_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm):
    """
    In -I'n (for ions).
    Modified Bessel function in the dielectric tensor.
    Eq. (10-57) of Stix book, p. 258

    We use I'n = (In-1 + In+1)/2.0
    """
    ll = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)
    In0 = ive(nharm, ll)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    Iprime = (In1 + In2) / 2.0
    return In0 - Iprime


@njit("float64(float64, float64, float64, float64, float64)")
def IminusIp_el(nperp, te_kev, Bmagn, freq, nharm):
    """
    In -I'n (for electrons).
    Modified Bessel function in the dielectric tensor.
    Eq. (10-57) of Stix book, p. 258

    We use I'n = (In-1 + In+1)/2.0
    """
    ll = lam_e(nperp, te_kev, Bmagn, freq)
    In0 = ive(nharm, ll)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    Iprime = (In1 + In2) / 2.0
    return In0 - Iprime


@njit("float64(float64, float64, float64, float64, float64, float64, float64)")
def nIn_z_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm):
    '''
    evaluate nIn/z needed in xx, yy, xz, zx components

    we use nIn/z = (In-1 - In+1)/2.0
    '''
    ll = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    return (In1 - In2) / 2.0


@njit("float64(float64, float64, float64, float64, float64)")
def nIn_z_el(nperp, te_kev, Bmagn, freq, nharm):
    '''
    evaluate nIn/z needed in xx, yy, xz, zx components

    we use nIn/z = (In-1 - In+1)/2.0
    '''
    ll = lam_e(nperp, te_kev, Bmagn, freq)
    In1 = ive(nharm - 1, ll)
    In2 = ive(nharm + 1, ll)
    return (In1 - In2) / 2.0


# Y MATRIX
@njit("complex128[:](float64, float64, float64, float64, float64, float64, float64, float64, float64)")
def chi_ions(nperp, npar, ni, A, Z, ti_kev, Bmagn, freq, nharm):
    """
    Chi matrix for ions
    """
    w_ci = wci(Bmagn, freq, A, Z)
    lam = lam_i(nperp, ti_kev, Bmagn, freq, A, Z)

    nIn_z = nIn_z_ions(nperp, ti_kev, Bmagn, freq,  A, Z,  nharm)
    I_Ip = IminusIp_ions(nperp, ti_kev, Bmagn, freq, A, Z, nharm)
    exp_In = ive(nharm, lam)

    vt = vti(ti_kev, A)
    zeta_i = 1.0 / (npar * vt) * (1.0 - nharm * w_ci)
    pl_z = zfunc(zeta_i)

    An = 1.0 / (om(freq) * npar * vt) * pl_z
    Bn = clight / (om(freq) * npar) * (1.0 + zeta_i * pl_z)

    wp2_fac = wpisq(ni, A, Z, freq) * om(freq)

    wp2An = An * wp2_fac
    wp2Bn = Bn * wp2_fac

    XX = nharm * nIn_z * wp2An
    XY = (-1j) * nharm * I_Ip * wp2An
    YY = XX + 2. * lam * I_Ip * wp2An

    XZ = nperp / w_ci / clight * nIn_z * wp2Bn
    YZ = (1j) * nperp / w_ci / clight * I_Ip * wp2Bn
    ZZ = 2.0 / clight / npar / vt**2 * (1.0 - nharm * w_ci) * exp_In * wp2Bn

    return np.array([XX, XY, YY, XZ, YZ, ZZ])


@njit("complex128[:](float64, float64, float64, float64, float64, float64, float64)")
def chi_el(nperp, npar, ne, te_kev, Bmagn, freq, nharm):
    """
    Chi matrix for electron
    """
    w_ce = wce(Bmagn, freq)
    lam = lam_e(nperp, te_kev, Bmagn, freq)

    nIn_z = nIn_z_el(nperp, te_kev, Bmagn, freq, nharm)
    I_Ip = IminusIp_el(nperp, te_kev, Bmagn, freq, nharm)
    exp_In = ive(nharm, lam)

    vt = vte(te_kev)
    zeta_e = 1.0 / (npar * vt) * (1.0 - nharm * w_ce)
    pl_z = zfunc(zeta_e)
    An = 1.0 / (om(freq) * npar * vt) * pl_z
    Bn = clight / (om(freq) * npar) * (1.0 + zeta_e * pl_z)

    wp2_fac = wpesq(ne, freq) * om(freq)
    wp2An = An * wp2_fac
    wp2Bn = Bn * wp2_fac

    XX = nharm * nIn_z * wp2An
    XY = (-1j) * nharm * I_Ip * wp2An
    YY = XX + 2. * lam * I_Ip * wp2An

    XZ = nperp / w_ce / clight * nIn_z * wp2Bn
    YZ = (1j) * nperp / w_ce / clight * I_Ip * wp2Bn
    ZZ = 2.0 / clight / npar / vt**2 * (1.0 - nharm * w_ce) * exp_In * wp2Bn

    return np.array([XX, XY, YY, XZ, YZ, ZZ])


@njit(complex128[:, ::1](float64, float64[:], float64[:], float64[:], darray_ro, iarray_ro,
                         float64, float64, float64, float64, int32))
def epsilonr_pl_hot_std(w, B, temps, denses, masses, charges, Te, ne, npara, nperp, nhrms):

    b_norm = sqrt(B[0]**2+B[1]**2+B[2]**2)
    freq = w/2/pi

    chi_terms = np.array([0j, 0j, 0j, 0j, 0j, 0j])

    if ne > 0.:
        te_kev = Te/1000.
        for nh in range(-nhrms, nhrms+1):
            chi_terms += chi_el(nperp, npara, ne, te_kev, b_norm, freq, nh)

    for Ti, dens, mass, charge in zip(temps, denses, masses, charges):
        ti_kev = Ti/1000.
        A = mass/Da
        Z = charge
        if dens <= 0.:
            continue
        for nh in range(-nhrms, nhrms+1):
            chi_terms += chi_ions(nperp, npara, dens, A,
                                  Z, ti_kev, b_norm, freq, nh)

    M = array([[1.+0j,   0., 0.],
               [0.,   1.+0j, 0.j],
               [0.,   0.,    1.+0j]])
    M2 = array([[chi_terms[0], chi_terms[1], chi_terms[3]],
                [-chi_terms[1], chi_terms[2], chi_terms[4]],
                [chi_terms[3], -chi_terms[4], chi_terms[5]], ])

    M += M2
    return M


@njit(complex128[:, :](float64[:], float64[:], complex128[:, :]))
def rotate_dielectric(B, K, M):
    #
    #  B : magnetic field.
    #  Kp : kperp
    #  M : dielectric matrix.
    #
    #

    B = ascontiguousarray(B)
    M = ascontiguousarray(M)
    K = ascontiguousarray(K)

    def rot_mat(ax, th):
        mat = array([[ax[0]**2*(1-cos(th))+cos(th),  ax[0]*ax[1]*(1-cos(th))-ax[2]*sin(th), ax[0]*ax[2]*(1-cos(th))+ax[1]*sin(th)],
                     [ax[0]*ax[1]*(1-cos(th))+ax[2]*sin(th), ax[1]**2*(1-cos(th)) +
                      cos(th),  ax[1]*ax[2]*(1-cos(th))-ax[0]*sin(th)],
                     [ax[0]*ax[2]*(1-cos(th))-ax[1]*sin(th), ax[1]*ax[2]*(1-cos(th))+ax[0]*sin(th), ax[2]**2*(1-cos(th))+cos(th)]],
                    dtype=complex128)
        return mat

    #  Step 1:
    #    algin ez to bn
    #    compute where ex goes after this step
    ez = array([0, 0, 1.0])
    ex = array([1.0, 0, 0.0], dtype=np.complex128)
    bn = B/sqrt(B[0]**2 + B[1]**2 + B[2]**2)
    ax = cross(bn, ez)
    # if bn // ez don't do anything
    if sqrt(sum(ax**2)) < 1e-16 and np.sum(bn*ez) > 0:
        ans2 = M
    else:
        if sqrt(sum(ax**2)) < 1e-16:
            ax = array([1.0, 0, 0.0])
            ay = array([0.0, 1.0, 0.0])
        else:
            ax = ax/sqrt(sum(ax**2))
            ay = cross(ax, bn)
        th = arctan2(sum(ez*ay), sum(ez*bn))
        mata = rot_mat(ax, th)
        matb = rot_mat(ax, -th)

        ans2 = dot(matb, dot(M, mata))
        ex = dot(matb, ex)

    # this is ex oriantation
    ex = ex.real
    ex = ex/sqrt(ex[0]**2 + ex[1]**2 + ex[2]**2)

    #  Step 2:
    #    Kperp is project of K on the plane perpendicular to bn
    K = K - K*bn

    #  Step 3
    #    algin ex to K

    ka = K/sqrt(K[0]**2 + K[1]**2 + K[2]**2)
    kb = cross(bn, ka)
    th = arctan2(sum(ex*kb), sum(ex*ka))

    mata = rot_mat(bn, th)
    matb = rot_mat(bn, -th)

    ans2 = dot(matb, dot(ans2, mata))
    #ex = dot(matb, ex)

    return ans2


@njit(complex128[:](float64[:], float64, float64[:], int64, complex128[:, :]))
def eval_npara_nperp(ptx, omega, kpakpe, kpe_mode, e_cold):

    if kpe_mode == 1:  # fast wave
        npara = speed_of_light*kpakpe[0]/omega
        S = e_cold[0, 0]
        D = e_cold[0, 1]*1j
        P = e_cold[2, 2]

        nperpsq = (D**2 - (npara**2 - S)**2)/(npara**2 - S)
        nperp = sqrt(nperpsq)
        #nperp = nperp.real
    elif kpe_mode == 2:  # slow wave
        npara = speed_of_light*kpakpe[0]/omega
        S = e_cold[0, 0]
        D = e_cold[0, 1]*1j
        P = e_cold[2, 2]
        nperpsq = -(npara**2 - S)*P/S
        nperp = sqrt(nperpsq)
        #nperp = nperp.real
    else:
        npara = speed_of_light*kpakpe[0]/omega
        nperp = speed_of_light*kpakpe[1]/omega

    return array([npara, nperp])

#
# routines to define kpe as vector
#


@njit(float64[:](float64[:], float64, float64, float64[:], float64[:]))
def eval_kpe_std(ptx, kpara, kperp, k, b):
    #
    #   kpe vector is given by k. it just project kpevec to a plane normal to
    #   b
    #
    bn = b/sqrt(b[0]**2 + b[1]**2 + b[2]**2)
    kn = k/sqrt(k[0]**2 + k[1]**2 + k[2]**2)
    tmp = cross(bn, kn)
    ret = cross(bn, tmp)
    # print(ret)
    return ret


@njit(float64[:](float64[:], float64, float64, float64[:], float64[:]))
def eval_kpe_em2da(ptx, kpara, kperp, k, b):
    #
    #   kvec specifies the direction of k on r-z plane
    #
    bn = b/sqrt(b[0]**2 + b[1]**2 + b[2]**2)

    ktor = -(k[0]*bn[0] + k[2]*bn[2])/bn[1]
    kvec = array([k[0], ktor, k[2]])

    #print('kvec', kvec)
    return kvec
