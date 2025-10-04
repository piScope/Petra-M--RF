from numba import njit, void, int32, int64, float64, complex128, types
from numpy import pi


from petram.phys.phys_const import epsilon0 as e0
from petram.phys.phys_const import q0 as q_base
from petram.phys.phys_const import c_cgs as clight

from petram.phys.phys_const import (q0_cgs,
                                    mass_electron,
                                    mass_proton)

gausspertesla = 1.E4
meter3_per_cm3 = 1.0E-6
me_gram = mass_electron * 1000.
mp_gram = mass_proton * 1000.


@njit("float64(float64)")
def om(freq):
    return 2.0 * pi * freq


@njit("float64(float64, float64)")
def wpesq(ne, freq):
    # wpe**2 normalized by omega**2
    return 4.0 * pi * ne * meter3_per_cm3 * q0_cgs**2 / me_gram / om(freq)**2


@njit("float64(float64, float64, float64, float64)")
def wpisq(ni, A, Z, freq):
    # wpi**2 normalized by omega**2

    return 4.0 * pi * ni * meter3_per_cm3 * \
        (q0_cgs * Z)**2 / (A * mp_gram) / om(freq)**2


@njit("float64(float64, float64)")
def wce(Bmagn, freq):
    # wce normalized by omega

    return (-q0_cgs * Bmagn * gausspertesla) / (me_gram * clight) / om(freq)


@njit("float64(float64, float64, float64, float64)")
def wci(Bmagn, freq, A, Z):
    # wci normalized by omega

    return (q0_cgs * Z * Bmagn * gausspertesla) / (A * mp_gram * clight) / om(freq)
