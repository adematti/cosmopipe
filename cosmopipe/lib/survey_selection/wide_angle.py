import logging

import numpy as np

from cosmopipe.lib.theory import ProjectionBase
from cosmopipe.lib.data_vector import ProjectionNameCollection
from .base import BaseRegularMatrix


def odd_wide_angle_coefficients(ell, wa_order=1, los='firspoint'):
    if wa_order != 1:
        raise NotImplementedError('Only wide-angle order 1 supported')

    def coefficient(ell):
        return ell*(ell+1)/2./(2*ell+1)

    sign = (-1)**(los == 'endpoint')
    if ell == 1:
        return [ell + 1], [sign * coefficient(ell+1)]
    return [ell-1, ell+1], [- sign * coefficient(ell-1), sign * coefficient(ell+1)]


class PowerOddWideAngle(BaseRegularMatrix):

    logger = logging.getLogger('PowerOddWideAngle')

    base = ProjectionBase(space=ProjectionBase.POWER,mode=ProjectionBase.MULTIPOLE)

    def __init__(self, d=1., wa_orders=1, los='firstpoint', sum_wa=True):
        self.d = d
        self.wa_orders = wa_orders
        if np.ndim(wa_orders) == 0:
            self.wa_orders = [wa_orders]
        if not np.allclose(self.wa_orders,[1]):
            raise NotImplementedError('Only wide-angle order wa_order = 1 supported')
        self.los = los
        self.sum_wa = sum_wa

    @property
    def k(self):
        return self.x

    def setup(self, k, projsin, projsout=None):
        self.projsin = ProjectionNameCollection(projsin)
        if projsout is None:
            self.projsout = self.propose_out(projsin)[-1]
        else:
            self.projsout = ProjectionNameCollection(projsout)
        if any(proj.wa_order is None for proj in self.projsin):
            raise ValueError('Input projections must have wide-angle order wa_order specified')
        if not self.sum_wa and any(proj.wa_order is None for proj in self.projsout):
            raise ValueError('Output projections must have wide-angle order wa_order specified')
        self.x = np.asarray(k)
        eye = np.eye(len(self.k),dtype=self.k.dtype)
        matrix = []
        for projout in self.projsout:
            line = []
            sum_wa = self.sum_wa and projout.wa_order is None
            if projout.proj % 2 == 0:
                for projin in self.projsin:
                    if projin.proj == projout.proj and (sum_wa or projin.wa_order == projout.wa_order):
                        tmp = eye
                    else:
                        tmp = 0.*eye
                    line.append(tmp)
            else:
                line = [0.*eye for projin in self.projsin]
                if sum_wa:
                    wa_orders = self.wa_orders
                else:
                    wa_orders = [projout.wa_order]
                for wa_order in wa_orders:
                    ells,coeffs = odd_wide_angle_coefficients(projout.proj,wa_order=wa_order,los=self.los)
                    for iprojin,projin in enumerate(self.projsin):
                        if projin.wa_order == 0 and projin.proj in ells:
                            coeff = coeffs[ells.index(projin.proj)]/self.d
                            if projin.proj == projout.proj + 1:
                                coeff_spherical_bessel = projin.proj + 1
                            else:
                                coeff_spherical_bessel = -projin.proj
                            #print(ell,ellout,coeff)
                            # K 'diag' terms in 3.3, 3.4 and 3.5
                            tmp = np.diag(coeff_spherical_bessel * coeff / self.k)
                            deltak = 2. * np.diff(self.k)
                            tmp += np.diag(coeff / deltak, k=1) - np.diag(coeff / deltak, k=-1)

                            tmp[0,0] -= 2.*coeff / deltak[0]
                            tmp[0,1] = 2.*coeff / deltak[0]
                            tmp[-1,-1] += 2.*coeff / deltak[-1]
                            tmp[-1,-2] = -2.*coeff / deltak[-1]
                            line[iprojin] += tmp
            matrix.append(line)
        """
        for projout in self.projsout:
            line = []
            for projin in self.projsin:
                if projin == projout:
                    tmp = np.eye(len(self.k),dtype=self.k.dtype)
                else:
                    tmp = 0.*np.eye(len(self.k),dtype=self.k.dtype)
                line.append(tmp)
            matrix.append(line)
        """
        #print([[tmp.shape for tmp in line] for line in matrix])
        self.matrix = np.bmat(matrix).A # (out,in)

    def propose_out(self, projsin):
        projsin = ProjectionNameCollection(projsin).select(**self.base.as_dict(drop_none=True))
        #projsin = ProjectionNameCollection([proj.copy(wa_order=proj.wa_order or 0) for proj in projsin if proj.wa_order is None or proj.wa_order % 2 == 0])
        projsin = ProjectionNameCollection([proj for proj in projsin if proj.wa_order is not None])
        ellsin = [proj.proj for proj in projsin if proj.wa_order is not None and proj.wa_order % 2 == 0]

        projsout = projsin.copy()
        for wa_order in self.wa_orders:
            for ellout in range(1,max(ellsin),2):
                if all(ell in ellsin for ell in odd_wide_angle_coefficients(ellout,wa_order=wa_order)[0]):
                    projsout.set(projsin[0].copy(wa_order=wa_order,proj=ellout))

        return projsin, projsout
