import math
import numpy as np

from matplotlib import pyplot as plt

from cosmopipe.lib.theory import PkEHNoWiggle, PkLinear
import pyccl

# Ref taken from https://github.com/Samreay/Barry/blob/master/barry/cosmology/power_spectrum_smoothing.py
# Compute the Eisenstein and Hu dewiggled transfer function
def __EH98_dewiggled(ks, om, ob, h0, rs):

    if rs == None:
        rs = __EH98_rs(om, ob, h0)

    # Fitting parameters
    a1 = 0.328
    a2 = 431.0
    a3 = 0.380
    a4 = 22.30
    g1 = 0.43
    g2 = 4.0
    c1 = 14.2
    c2 = 731.0
    c3 = 62.5
    l1 = 2.0
    l2 = 1.8
    t1 = 2.0
    theta = 2.725 / 2.7  # Normalised CMB temperature

    q0 = ks * theta * theta
    alpha = 1.0 - a1 * math.log(a2 * om * h0 * h0) * (ob / om) + a3 * math.log(a4 * om * h0 * h0) * (ob / om) ** 2
    gamma_p1 = (1.0 - alpha) / (1.0 + (g1 * ks * rs * h0) ** g2)
    gamma = om * h0 * (alpha + gamma_p1)
    q = q0 / gamma
    c = c1 + c2 / (1.0 + c3 * q)
    l = np.log(l1 * math.exp(1.0) + l2 * q)
    t = l / (l + c * q ** t1)

    return t


def __EH98_lnlike(params, ks, pkEH, pkdata):

    pk_B, pk_a1, pk_a2, pk_a3, pk_a4, pk_a5 = params

    Apoly = pk_a1 * ks + pk_a2 + pk_a3 / ks + pk_a4 / ks ** 2 + pk_a5 / ks ** 3
    pkfit = pk_B * pkEH + Apoly

    # Compute the chi_squared
    chi_squared = np.sum(((pkdata - pkfit) / pkdata) ** 2)

    return chi_squared


def __sigma8_integrand(ks, kmin, kmax, pkspline):
    if (ks < kmin) or (ks > kmax):
        pk = 0.0
    else:
        pk = interpolate.splev(ks, pkspline, der=0)
    window = 3.0 * ((math.sin(8.0 * ks) / (8.0 * ks) ** 3) - (math.cos(8.0 * ks) / (8.0 * ks) ** 2))
    return ks * ks * window * window * pk


# Compute the Eisenstein and Hu 1998 value for the sound horizon
def __EH98_rs(om, ob, h0):

    # Fitting parameters
    b1 = 0.313
    b2 = -0.419
    b3 = 0.607
    b4 = 0.674
    b5 = 0.238
    b6 = 0.223
    a1 = 1291.0
    a2 = 0.251
    a3 = 0.659
    a4 = 0.828
    theta = 2.725 / 2.7  # Normalised CMB temperature

    obh2 = ob * h0 * h0
    omh2 = om * h0 * h0

    z_eq = 2.5e4 * omh2 / (theta ** 4)
    k_eq = 7.46e-2 * omh2 / (theta ** 2)

    zd1 = b1 * omh2 ** b2 * (1.0 + b3 * omh2 ** b4)
    zd2 = b5 * omh2 ** b6
    z_d = a1 * (omh2 ** a2 / (1.0 + a3 * omh2 ** a4)) * (1.0 + zd1 * obh2 ** zd2)

    R_eq = 3.15e4 * obh2 / (z_eq * theta ** 4)
    R_d = 3.15e4 * obh2 / (z_d * theta ** 4)

    s = 2.0 / (3.0 * k_eq) * math.sqrt(6.0 / R_eq) * math.log((math.sqrt(1.0 + R_d) + math.sqrt(R_d + R_eq)) / (1.0 + math.sqrt(R_eq)))

    return s


def test_eh():

    cosmo = dict(Omega_c=0.25,Omega_b=0.05,h=0.7,n_s=0.95,sigma8=0.8)
    k = np.logspace(-4.,0.,100)
    pknow = PkEHNoWiggle(k=k)

    tref = __EH98_dewiggled(k, cosmo['Omega_c'] + cosmo['Omega_b'], cosmo['Omega_b'], cosmo['h'], None)
    tnow = pknow.transfer(h=cosmo['h'], Omega_b=cosmo['Omega_b'], Omega_c=cosmo['Omega_c'], T_cmb=2.725)
    assert np.allclose(tnow,tref,rtol=2e-3,atol=0)
    plt.plot(k,tref/tnow)
    plt.xscale('log')
    plt.show()

def test_boltzmann():
    cosmo_params = dict(Omega_c=0.25,Omega_b=0.05,h=0.7,n_s=0.95,sigma8=0.8)
    cosmo = pyccl.Cosmology(**cosmo_params,transfer_function='boltzmann_class')
    from pyccl import ccllib
    nk = ccllib.get_pk_spline_nk(cosmo.cosmo)
    kmax = cosmo.cosmo.spline_params.K_MAX_SPLINE
    kmin = 1e-5
    a = 1.
    k = np.logspace(np.log10(kmin),np.log10(kmax),nk)/cosmo['h']
    def pk_callable(k):
        return cosmo['h']**3*pyccl.linear_matter_power(cosmo,cosmo['h']*k,a=a)
    pk = pk_callable(k)

    cosmoeh = pyccl.Cosmology(**cosmo_params,transfer_function='eisenstein_hu')
    pkeh = cosmo['h']**3*pyccl.linear_matter_power(cosmoeh,cosmo['h']*k,a=a)

    pknow = PkEHNoWiggle(k=k)
    pknow.run(**cosmo_params)
    pknowadjust = pknow.deepcopy()
    pklin = PkLinear.from_callable(k,pk_callable)
    pknowadjust.adjust_to_pk(pklin)
    pknowadjustbarry = pknow.deepcopy()
    pknowadjustbarry.adjust_to_pk_barry(pklin)

    plt.plot(k,pk/pkeh,label='CLASS/EH')
    plt.plot(k,pk/pknow['pk'],label='CLASS/EHnoW')
    plt.plot(k,pk/pknowadjust['pk'],label='CLASS/EHnoW + Polyfit')
    plt.plot(k,pk/pknowadjustbarry['pk'],label='CLASS/EHnoW + Polyfit Barray')
    plt.xlim(1e-4,0.5)
    plt.yscale('linear')
    plt.legend()
    plt.show()

def test_background():
    cosmo_params = dict(Omega_c=0.25,Omega_b=0.05,h=0.7,n_s=0.95,sigma8=0.8)
    cosmo1 = pyccl.Cosmology(**cosmo_params,transfer_function='boltzmann_class')
    cosmo_params = dict(Omega_c=0.25,Omega_b=0.05,h=0.8,n_s=0.95,sigma8=0.8)
    cosmo2 = pyccl.Cosmology(**cosmo_params,transfer_function='boltzmann_class')
    a = 0.5
    assert np.allclose(pyccl.angular_diameter_distance(cosmo1,a)*cosmo1['h'],pyccl.angular_diameter_distance(cosmo2,a)*cosmo2['h'],rtol=1e-4)
    assert np.allclose(pyccl.h_over_h0(cosmo1,a),pyccl.h_over_h0(cosmo2,a),rtol=1e-4)

    from classy import Class
    cls = Class()
    cosmo_params = dict(Omega_cdm=0.25,Omega_b=0.05,h=0.7,n_s=0.95)
    cosmo_params = {'omega_cdm': 0.12239665944373529, 'omega_b': 0.02540909039244506, 'h': 0.6818164667333406, 'n_s': 0.96}
    cls.set(**cosmo_params)
    print(cls.rs_drag())


if __name__ == '__main__':
    #test_eh()
    #test_boltzmann()
    test_background()
