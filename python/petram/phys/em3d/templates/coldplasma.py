from numpy import pi, sin, cos, exp, sqrt, log, arctan2
# constants    
w = 4.6e9 * 2* pi           # omega

e0 = 8.8541878176e-12       # vacuum permittivity

Zi = 1
Zim = 18

q = 1.60217662e-19
qe = q
qi = Zi*q
qim = Zim*q
me = 9.10938356e-31
u = 1.660539040e−27  # unified mass unit
mi = 1.00794*u
mi = 2.01410178*u    # Dutrium
mim = 39.948*u       # Ar (39.948)

imp_frac = 0.01

#poloidal field
from scipy.interpolate import interp2d
Brfit =  interp2d(rgrid, zgrid, brrz)
Bzfit =  interp2d(rgrid, zgrid, btrz)
Btfit =  interp2d(rgrid, zgrid, bzrz)

def Bfields(x, y, z):
    return Brfit([x, y, z]), Bzfit([x, y, z]), Btfit([x, y, z])

def B(x, y, z):
    Br, Bz, Bt = Bfield(x, y, z)
    return sqrt(Br**2+Bz**2+Bt**2)

def ph_th(x, y, z):
    Br, Bz, Bt = Bfield(x, y, z)
    th = arctan2(Bz, Br)
    ph = arctan2(Br*cos(th)+Bz*sin(th),Bt)
    return ph, th

def ne(x, y, z):
    return 1e19
def ni(x, y, z):
    return ne(x, y, z) *(1. - imp_frac)
def nim(x, y, z):
    return ne(x, y, z) * imp_frac)

def Te(x, y, z):
    return 100*q     # 100eV in MKSA 

def Ti(x, y, z):
    return 100*q     # 100eV in MKSA 

def Tim(x, y, z):
    return 100*q     # 100eV in MKSA 

def vTe(x, y, z):
    return sqrt(2*Te(x, y, z)/me)
def vTi(x, y, z):
    return sqrt(2*Ti(x, y, z)/mi)
def vTim(x, y, z):
    return sqrt(2*Tim(x, y, z)/mim)

def LAMBDA(x, y, z):
    # 1+LAMDA as in log(1+LAMBDA) coulomb logarithm
    Te, ne = Te(x, y, z), ne(x, y, z)
    1+12*pi*(e0*Te)**(3./2)/(q**3 * sqrt(ne))

def nu_ei(x, y, z):
    return (qi**2 * qe**2 * ni(x, y, z) *
            log(LAMBDA(x, y, z))/(4 * pi*e0**2*me**2)/vTe(x, y, z)^3)

def nu_eim(x, y, z):
    return (qim**2 * qe**2 * nim(x, y, z) *
            log(LAMBDA(x, y, z))/(4 * pi*e0**2*me**2)/vTe(x, y, z)^3)

def me_eff(x, y z):
    #effective electrons mass (to account for collisions)
    return (1+nu_ei(x, y, z) / 1j/w + nu_eim(x, y, z)/1j/w )*me

def mi_eff(x, y z):
    #effective ion mass (to account for collisions)
    return (1+nu_ei(x, y, z) / 1j/w + nu_eim(x, y, z)/1j/w )*mi

def mim_eff(x, y z):
    #effective ion mass (to account for collisions)
    return (1+nu_ei(x, y, z) / 1j/w + nu_eim(x, y, z)/1j/w )*mim

def wp(x, y, z):
    # electron plasma freuquency
    return sqrt(ne(x, y, z) * qe**2/(me_eff(x, y, z)*e0))

def wpi(x, y, z):
    #sqrt(ni*qi^2/(mi_eff*e0)) ion plasma frequency        
    return sqrt(ni(x, y, z) * qi**2/(mi_eff(x, y, z)*e0))

def wpim(x, y, z):
    # electron plasma freuquency

    return sqrt(nim(x, y, z) * qim**2/(mim_eff(x, y, z)*e0))

def wce(x, y, z):
    # electron cyclotron frequency
    return -q * B(x, y, z)/me_eff(x, y, z)
def wci(x, y, z):
    # ion cyclotron frequency
    return qi * B(x, y, z)/mi_eff(x, y, z)
def wce(x, y, z)
    return qim * B(x, y, z)/mim_eff(x, y, z)

def wc_wp(x, y, z):
    return (wpe(x, y, z), wpi(x, y, z), wpim(x, y, z),
            wce(x, y, z), wci(x, y, z), wcim(x, y, z),)

def P_rf(x, y, z):
    #STIX P
    wpe, wpi, wpim, wce, wci, wcim = wc_wp(x, y, z)    
    return (1 - wpe**2/w**2
              - wpi**2/w**2
              - wpim**2/w**2)
def S_rf(x, y, z):
    #STIX S    
    wpe, wpi, wpim, wce, wci, wcim = wc_wp(x, y, z)
    return 1-wpe**2/(w**2-wce**2)-wpi**2/(w**2-wci**2)-wpim**2/(w**2-wcim**2)

def D_rf(x, y, z):
    # STIX D
    wpe, wpi, wpim, wce, wci, wcim = wc_wp(x, y, z)    
    return (wce*wpe**2/(w*(w**2-wce**2)) + wci*wpi**2/(w*(w**2-wci**2)) +
            wcim*wpim**2/(w*(w**2-wcim**2)))

def Stix(x, y, z):
    return S_rf(x, y, z), D_rf(x, y, z), P_rf(x, y, z)

def e_xx(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return (- P*sin(ph)**2*sin(th)**2
            + P*sin(ph)**2 + S*sin(ph)**2 * sin(th)**2
            - S*sin(ph)**2 + S)
def e_xy(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return (1j*D*sin(th) + P*cos(ph)*cos(th) - S*cos(ph)*cos(th))*sin(ph)

def e_xz(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return -1j*D*cos(ph) + P*sin(ph)**2*sin(th)*cos(th)-S*sin(ph)**2*sin(th)*cos(th)

def e_yx(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return  -(1j*D*sin(th)-P*cos(ph)*cos(th)+S*cos(ph)*cos(th))*sin(ph)

def e_yy(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return P*cos(ph)**2 + S*sin(ph)**2

def e_yz(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return  (1j*D*cos(th)+P*sin(th)*cos(ph)-S*sin(th)*cos(ph))*sin(ph) 

def e_zx(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return 1j*D*cos(ph) + P*sin(ph)**2*sin(th)*cos(th) - S*sin(ph)**2*sin(th)*cos(th)

def e_zy(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return -(1j*D*cos(th)-P*sin(th)*cos(ph)+S*sin(th)*cos(ph))*sin(ph) 

def e_zz(x, y, z):
    ph, th = ph_th(x, y, z)
    S, D, P = Stix(x, y, z)    
    return P*sin(ph)**2*sin(th)**2 - S*sin(ph)**2*sin(th)**2 + S



'''
S S_rf
P P_rf
D D_rf
Ezeta (emw.Er*Br+emw.Ez*Bz+emw.Ephi*Bt)/B
Eeta (emw.Er*Br+emw.Ez*Bz)/Bp*cos(ph)-emw.Ephi*sin(ph)
Bzeta (emw.Br*Br+emw.Bz*Bz+emw.Bphi*Bt)/B
Beta (emw.Br*Br+emw.Bz*Bz)/Bp*cos(ph)-emw.Bphi*sin(ph)

Eb_phi Eb*exp(i*pmode*theta(r,z))*cos(ph+e_angle) not used
Eb_z Eb*exp(i*pmode*theta(r,z))*sin(ph+e_angle)*sin(th) not used
Eb_r Eb*exp(i*pmode*theta(r,z))*sin(ph+e_angle)*cos(th) not used
Bt_toric -Bt
Bp_toric s_toric*Bp
Theta atan2(Bp_toric,Bt_toric) Capital theta in Toric
Eeta_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*s_toric*cos(Thetab(r,z))-sin(Thetab(r,z))*(-emw.Ephi)
Beta_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*s_toric*cos(Thetab(r,z))-sin(Thetab(r,z))*(-emw.Bphi)
Ezeta_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*s_toric*sin(Thetab(r,z))+cos(Thetab(r,z))*(-emw.Ephi)
Bzeta_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*s_toric*sin(Thetab(r,z))+cos(Thetab(r,z))*(-emw.Bphi)
Epsi (emw.Er*Br+emw.Ez*Bz)/Bp*sin(ph)+emw.Ephi*cos(ph)
tEr_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*cos(th_efit(r,z))
tEz_b (emw.Er*cos(th_efit(r,z))+emw.Ez*sin(th_efit(r,z)))*sin(th_efit(r,z))
tBr_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*cos(th_efit(r,z))
tBz_b (emw.Br*cos(th_efit(r,z))+emw.Bz*sin(th_efit(r,z)))*sin(th_efit(r,z))
'''
