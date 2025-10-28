'''

 routines used for plasma waves

'''
from petram.phys.phys_const import q0
from petram.phys.phys_const import q0_cgs as qe
from petram.phys.phys_const import (mass_electron, mass_proton)
from petram.phys.phys_const import c_cgs as clight

# CONSTANTS (this is original)
gausspertesla = 1.E4
meter3_per_cm3 = 1.0E-6


me = mass_electron * 1000.
mp = mass_proton * 1000.
ergperkev = q0 * 1e10

from numpy import pi, sqrt

@njit("float64(float64)")
def om(freq):
    return 2.0 * pi * freq

@njit("float64(float64, float64)")
def wpesq(ne, freq):
    return 4.0 * pi * ne * meter3_per_cm3 * qe**2 / me / om(freq)**2

@njit("float64(float64, float64, float64, float64)")
def wpisq(ni, A, Z, freq):
    return 4.0 * pi * ni * meter3_per_cm3 * \
        (qe * Z)**2 / (A * mp) / om(freq)**2

@njit("float64(float64, float64)")
def wce(Bmagn, freq):
    return (-qe * Bmagn * gausspertesla) / (me * clight) / om(freq)

@njit("float64(float64, float64, float64, float64)")
def wci(Bmagn, freq, A, Z):
    return (qe * Z * Bmagn * gausspertesla) / (A * mp * clight) / om(freq)

@njit("float64(float64)")
def vte(te_kev):
    tee = ergperkev * te_kev
    return sqrt(2.0 * tee / me) / clight

@njit("float64(float64, float64)")
def vti(ti_kev, A):
    tii = ergperkev * ti_kev    
    return sqrt(2.0 * tii / (A * mp)) / clight

@njit("float64(float64, float64, float64, float64, float64)")
def lam_i(nperp, ti_kev, Bmagn, freq, A, Z):
    """Lambda for ions in Stix notation (Eq. (10-55) Stix's book, p. 258 ).
    """
    return nperp**2 * vti(ti_kev, A)**2 / (2.0 * wci(Bmagn, freq, A, Z)**2)

@njit("float64(float64, float64, float64, float64)")
def lam_e(nperp, te_kev, Bmagn, freq):
    """Lambda for electrons in Stix notation (Eq. (10-55) Stix's book, p. 258 ).
    """
    return nperp**2 * vte(te_kev)**2 / (2.0 * wce(Bmagn, freq)**2)
