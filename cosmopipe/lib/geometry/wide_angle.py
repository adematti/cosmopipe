import logging

import numpy as np

from cosmopipe.lib.utils import BaseClass


def odd_wide_angle_coefficients(ell, n=1, los=0):
    if n != 1:
        raise NotImplementedError('Odd order n = 1 only is implementented')

    def coefficient(ell):
        return ell*(ell+1)/2./(2*ell+1)

    sign = (-1)**los
    if ell == 1:
        return [ell + 1], [sign * coefficient(ell+1)]
    return [ell-1, ell+1], [- sign * coefficient(ell-1), sign * coefficient(ell+1)]


class PowerOddWideAngle(BaseClass):

    def __init__(self, ells, k=None, d=1., n=1, ellsout=None, los=0):
        self.ells = ells
        self.ellsout = ellsout
        if self.ellsout is None:
            self.ellsout = list(range(1,max(ells),2))
        self.k = np.asarray(k)
        ells = np.array(ells)
        zeros = np.zeros((len(self.k),)*2,dtype=self.k.dtype)
        matrix = []
        for ellout in self.ellsout:
            line = []
            if ellout % 2 == 0:
                line += [zeros for ell in self.ells] # no even terms
            else:
                ells,coeffs = odd_wide_angle_coefficients(ellout,n=n,los=los)
                for ell in self.ells:
                    if ell in ells:
                        coeff = coeffs[ells.index(ell)]/d
                        if ell == ellout + 1:
                            coeff_spherical_bessel = ell + 1
                        else:
                            coeff_spherical_bessel = -ell
                        #print(ell,ellout,coeff)
                        # K 'diag' terms in 3.3, 3.4 and 3.5
                        tmp = np.diag(coeff_spherical_bessel * coeff / self.k)
                        deltak = 2. * np.diff(self.k)
                        tmp += np.diag(coeff / deltak, k=1) - np.diag(coeff / deltak, k=-1)

                        tmp[0,0] -= 2.*coeff / deltak[0]
                        tmp[0,1] = 2.*coeff / deltak[0]
                        tmp[-1,-1] += 2.*coeff / deltak[-1]
                        tmp[-1,-2] = -2.*coeff / deltak[-1]
                    else:
                        tmp = 0.*eye
                    line.append(tmp)
            matrix.append(line)
        matrix = np.array(matrix) # now (ellsout,ells,kout,k)
        self.matrix = np.transpose(matrix,(2,0,3,1)) # now (kout,ellsout,k,ells)

    def compute(self, func):
        return np.sum(self.matrix*func,axis=(2,3))

    def __getstate__(self, state):
        for key in ['ells','ellsout','k','matrix']:
            state[key] = getattr(self,key)
        return state
