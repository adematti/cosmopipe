import functools

import numpy as np
from scipy import interpolate

from cosmopipe.lib.utils import BaseClass
from .fog import get_FoG


def scale_factor(power):
    def decorate(func):
        @functools.wraps(func)
        def wrapper(self, scale=1., *args, **kwargs):
            return scale**power*func(self,*args,**kwargs)
        return wrapper
    return decorate


class PTQuantity(BaseClass):

    fields = []
    shapes = {}
    scale_powers = {}

    def __init__(self, data=None, **kwargs):
        if isinstance(data,PTQuantity):
            self.__dict__.update(data.__dict__)
            return
        data = data or {}
        data.update(kwargs)
        self.data = {field: data[field] for field in self.fields if field in data}

    def __len__(self):
        return len(self['k'])

    @property
    def size(self):
        return len(self)

    def __repr__(self):
        return 'PTQuantity({})'.format(self.__dict__)

    def __str__(self):
        return str(self.__dict__)

    def __getitem__(self, name):
        if isinstance(name,str):
            return self.data[name]
        return self.__class__(**{field:self.data[field][name] for field in self.fields})

    def __setitem__(self, name, item):
        if isinstance(name,str):
            self.data[name] = item
        else:
            for field in self.fields:
                self.data[field][name] = item

    def __contains__(self, name):
        return name in self.data

    def __iter__(self):
        return iter(self.fields)

    def tolist(self):
        return [self[field] for field in self]

    def pop(self,key):
        return self.data.pop(key)

    def as_dict(self,fields=None):
        if fields is None: fields = self.fields
        return {field:self[field] for field in fields}

    def deepcopy(self):
        new = self.__class__()
        for field in self.fields: new.data[field] = self.data[field].copy()
        return new

    def pad_k(self, k, mode='constant', constant_values=0., **kwargs):
        pad_start = np.sum(k<self.k[0])
        pad_end = np.sum(k>self.k[-1])
        for field in self.fields:
            pad_width = ((0,0))*(self[field].ndim-1) + ((pad_start,pad_end))
            self[field] = np.pad(self[field],pad_width=pad_width,mode=mode,constant_values=constant_values)
        self['k'][:pad_start] = k[:pad_start]
        self['k'][-pad_end:] = k[-pad_end:]
        for field in kwargs:
            self[field][:pad_start] = kwargs[field][:pad_start]
            self[field][-pad_end:] = kwargs[field][-pad_end:]

    def nan_to_zero(self):
        for field in self.fields: self[field][np.isnan(self[field])] = 0.

    def rescale(self, scale=1.):
        for field in self.fields: self[field] *= scale**(self.scale_powers[field])

    def zeros(self, k=None, dtype='f8'):
        if k is None:
            k = self.k
        self['k'] = np.asarray(k,dtype=dtype).flatten()
        for field in self.fields:
            if field == 'k': continue
            if field in self.shapes: self[field] = np.zeros((self.size,)+tuple(self.shapes[field]),dtype=dtype).flatten()
            else: self[field] = np.zeros((self.size,),dtype=dtype)

    def reshape(self):
        for field in self.fields:
            if field in self.shapes: self[field].shape = (self.size,) + self.shapes[field]

    @property
    def k(self):
        return self['k']

    def interp(self, k, field='pk', kind='cubic'):
        # NOTE: scipy.interpolate.interp1d(kind='linear') is about 4x slower than np.interp... and kind='cubic' again 4 times slower.
        return interpolate.interp1d(self.k,self[field],kind=kind,axis=-1,copy=False,bounds_error=True,assume_sorted=True)(k)

    __call__ = interp

    def sigmav(self, **kwargs):
        return np.sqrt(1./6./np.pi**2*np.trapz(self.pk(**kwargs),x=self.k,axis=-1))

    def sigmar(self, r, **kwargs):
        x = self.k*r
        w = 3.*(np.sin(x)-x*np.cos(x))/x**3
        sigmar2 = 1./2./np.pi**2*np.trapz(self.pk(**kwargs)*(w*self.k)**2,x=self.k,axis=-1)
        return np.sqrt(sigmar2)

    def sigma8(self, **kwargs):
        return self.sigmar(8., **kwargs)


class PkLinear(PTQuantity):

    fields = ['k','pk']
    scale_powers = {'k':0,'pk':2}

    @scale_factor(2)
    def pk(self):
        return self['pk']

    @classmethod
    def from_callable(cls, k, pk_callable):
        self = cls(k=k,pk=pk_callable(k))
        self.interp = pk_callable
        return self


class PkEHNoWiggle(PkLinear):

    fields = ['k','pk']
    scale_powers = {'k':0,'pk':2}

    def transfer(self, h=0.676, Omega_b=0.05, Omega_c=0.25, T_cmb=2.7255):
        # Fitting formula for no-wiggle P(k) (Eq. [29] of Eisenstein and Hu 1998)
        omega_m = (Omega_b + Omega_c) * h**2
        omega_b = Omega_b * h**2
        frac_baryon  = omega_b / omega_m
        theta_cmb = T_cmb / 2.7
        sound_horizon = 44.5 * h * np.log(9.83/omega_m) / np.sqrt(1. + 10.*omega_b**0.75)
        alpha_gamma = 1. - 0.328 * np.log(431.*omega_m) * frac_baryon + 0.38 * np.log(22.3*omega_m) * frac_baryon**2
        ks = self.k * sound_horizon
        gamma_eff = omega_m / h * (alpha_gamma + (1 - alpha_gamma) / (1 + (0.43*ks) ** 4))
        q = self.k * theta_cmb**2 / gamma_eff
        L0 = np.log(2*np.e + 1.8 * q)
        C0 = 14.2 + 731.0 / (1 + 62.5 * q)
        return L0 / (L0 + C0 * q**2)

    def run(self, k=None, sigma8=None, n_s=1., **kwargs):
        self.zeros(k=k)
        self['pk'] = self.k**n_s*self.transfer(**kwargs)**2
        if sigma8 is not None: current_sigma8 = self.sigma8()
        self['pk'] *= (sigma8/current_sigma8)**2

    def adjust_to_pk(self, pk, kfit=None):
        from scipy import optimize
        if kfit is None: kfit = self.k
        pkref = pk(kfit)
        pknow = self(kfit)

        def model(k, ap0, ap1, ap2, ap3, ap4):
            return pkref/(pknow * (ap0 + ap1*k + ap2*k**2 + ap3*k**3 + ap4*k**4))

        popt, pcov = optimize.curve_fit(model,kfit,np.ones_like(kfit),p0=[1.]+[0.]*4,maxfev=100000)
        k = self.k
        pkref = pk(k)
        pknow = self(k)
        wiggles = model(k,*popt)
        kmin,kmax = kfit[0],kfit[-1]
        ones = np.ones_like(k)
        mask = k>kmax
        ones[mask] *= np.exp(-1e3*(k[mask]/kmax-1)**2)
        mask = k<kmin
        ones[mask] *= np.exp(-(kmin/k[mask]-1)**2)
        wiggles = (wiggles-1.)*ones + 1.
        self['pk'] = pkref/wiggles

    def adjust_to_pk_barry(self, pk, kfit=None):
        from scipy import optimize
        if kfit is None: kfit = self.k
        #def model(k, ap0, ap1, ap2, ap3, ap4):
        #    return pkref/(pknow * (ap0 + ap1*k + ap2*k**2 + ap3*k**3 + ap4*k**4))
        def model(pars, k, pknow):
            b, ap0, ap1, am1, am2, am3 = pars
            return b*(pknow + ap0 + ap1*k + am1/k + am2/k**2 + am3/k**3)

        def chi2(pars, k, pknow, pklin):
            return np.sum(((pklin - model(pars,k,pknow))/pklin)**2)

        x0 = np.array([1.]+[0.]*5)
        result = optimize.minimize(chi2,x0=x0,args=(kfit,self(kfit),pk(kfit)),method="Nelder-Mead",tol=1.0e-6,options={"maxiter": 1000000})

        self['pk'] = model(result.x,self.k,self['pk'])



class BasePTModel(BaseClass):

    def __init__(self, pklin, klin=None, FoG='gaussian', cosmo=None):
        if klin is None:
            if callable(pklin):
                self.pk_linear = pklin
            elif isinstance(pklin,PkLinear):
                self.pk_linear = pklin.copy()
            else:
                raise ValueError('Input pklin should be a PkLinear if no k provided.')
        else:
            self.pk_linear = PkLinear({'k':klin,'pk':pklin})
        self.FoG = FoG
        if FoG is not None:
            self.FoG = get_FoG(FoG)
        self.cosmo = cosmo or {}
