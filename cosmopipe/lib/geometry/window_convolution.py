import math

import numpy as np


class CorrelationWindowMatrix(BaseClass):

    logger = logging.getLogger('CorrelationWindowMatrix')

    def __init__(self, ells, ellsout=None, window=None, s=None, **kwargs):
        if ellsout is None: ellsout = ells
        self.ells, self.ellsout = ells, ellsout
        self.s = s
        tmp = window(s,ell=0,**kwargs)
        matrix = np.empty(tmp.shape + (len(self.ells),len(self.ellsout)),dtype=tmp.dtype)
        for illout,ellout in enumerate(self.ellsout):
            for illin,ellin in enumerate(self.ells):
                ells,coeffs = wigner3j_square(ellout,ellin) #case ellin = (ell,n)
                matrix[illout][illin] = np.sum([coeff*window(s,ell=ell) for ell,coeff in zip(ells,coeffs)],axis=0)
        self.matrix = np.transpose(matrix,(2,1,0)) # matrix is now (s,ells,ellsout)

    def compute(self, func):
        # matrix is (ellsout,ells,s), func is (s,ells)
        return np.sum(self.matrix*func,axis=1)

    def __getstate__(self, state):
        for key in ['ells','ellsout','s','matrix']:
            state[key] = getattr(self,key)
        return state


class PowerWindowMatrix(CorrelationWindowMatrix):

    logger = logging.getLogger('PowerWindowMatrix')

    def __init__(self, ells, ellsout=None, window=None, kout=None, krange=None, srange=(1e-4,1e4), ns=1024*64, q=0, **kwargs):
        if krange is not None:
            srange = (1./krange[1],1./krange[0])
        self.kout = kout
        s = np.logspace(np.log10(srange[0]),np.log10(srange[1]),ns)
        super(CorrelationWindowMatrix,self).__init__(ells=ells,ellsout=ellsout,window=window,s=s,**kwargs)
        fflog = CorrelationToPower(s,ell=ells,q=q,lowring=False)
        k, matrix = fflog(np.transpose(self.matrix,(2,1,0))) # now (ellsout,ells,k)
        self.k = k[0]
        prefactor = 2./np.pi * ((1j)**np.array(self.ellsout))[:,None] * ((-1j)**np.array(self.ells))[None,:]
        # now (ellsout, kout, ells, k)
        matrix = (prefactor * matrix)[:,None,...] * np.array([special.spherical_jn(ell,s[None,:]*self.kout[:,None]) for ellout in ellsout])[:,:,None,:]
        self.matrix = np.transpose(matrix.real,(1,0,3,2)) # now (kout,ellsout,k,ells)

    def compute(self, func):
        # matrix is (kout,ellsout,k,ells), func is (k,ells)
        return np.sum(self.matrix*func,axis=(2,3))

    def __getstate__(self, state):
        for key in ['ells','ellsout','k','kout','matrix']:
            state[key] = getattr(self,key)
        return state


def wigner3j_square(ellout, ellin, prefactor=True, as_string=False):

    coeffs = []
    qvals = []
    retstr = []

    def G(p):
        """Return the function G(p), as defined in Wilson et al 2015.
        See also: WA Al-Salam 1953
        Taken from https://github.com/nickhand/pyRSD.

        Returns
        -------
        numer, denom: int
            the numerator and denominator

        """
        toret = 1
        for p in range(1,p+1): toret *= (2*p - 1)
        return toret, math.factorial(p)

    for p in range(min(ellin,ellout)+1):

        numer, denom = [], []

        # numerator of product of G(x)
        for r in [G(ellout-p), G(p), G(ellin-p)]:
            numer.append(r[0])
            denom.append(r[1])

        # divide by this
        a,b = G(ellin+ellout-p)
        numer.append(b)
        denom.append(a)

        numer.append(2*(ellin+ellout) - 4*p + 1)
        denom.append(2*(ellin+ellout) - 2*p + 1)

        q = ellin + ellout - 2*p
        if prefactor:
            numer.append(2*ellout + 1)
            denom.append(2*q + 1)

        numer = Fraction(np.prod(numer))
        denom = Fraction(np.prod(denom))
        if not as_string:
            coeffs.append(numer*1./denom)
            qvals.append(q)
        else:
            retstr.append('l{:d} {}'.format(q,numer/denom))

    if not as_string:
        return qvals[::-1], coeffs[::-1]
    return retstr[::-1]
