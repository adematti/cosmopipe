"""Implementation of wide-angle expansion."""

import logging

import numpy as np

from cosmopipe.lib.theory import ProjectionBasis
from cosmopipe.lib.data_vector import ProjectionNameCollection
from .base import BaseRegularMatrix


def odd_wide_angle_coefficients(ell, wa_order=1, los='firspoint'):
    r"""
    Compute coefficients of odd wide-angle expansion, i.e.:

    .. math::

        \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right)}, - \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right)}

    See https://fr.overleaf.com/read/hpgbwqzmtcxn.
    A minus sign is applied on both factors if ``los`` is 'endpoint'.

    Parameters
    ----------
    ell : int
        Multipole order.

    wa_order : int, default=1
        Wide-angle expansion order.
        So far only order 1 is supported.

    los : string
        Choice of line-of-sight, either:

        - 'firstpoint': the separation vector starts at the end of the line-of-sight
        - 'endpoint': the separation vector ends at the end of the line-of-sight.

    Returns
    -------
    ells : list
        List of multipole orders of correlation function.

    coeffs : list
        List of coefficients to apply to correlation function multipoles corresponding to output ``ells``.
    """
    if wa_order != 1:
        raise NotImplementedError('Only wide-angle order 1 supported')

    def coefficient(ell):
        return ell*(ell+1)/2./(2*ell+1)

    sign = (-1)**(los == 'endpoint')
    if ell == 1:
        return [ell + 1], [sign * coefficient(ell+1)]
    return [ell-1, ell+1], [- sign * coefficient(ell-1), sign * coefficient(ell+1)]


class PowerOddWideAngle(BaseRegularMatrix):
    """
    Class computing matrix for wide-angle expansion.
    Adapted from https://github.com/fbeutler/pk_tools/blob/master/wide_angle_tools.py
    """
    basis = ProjectionBasis(space=ProjectionBasis.POWER,mode=ProjectionBasis.MULTIPOLE)

    def __init__(self, d=1., wa_orders=1, los='firstpoint', sum_wa=True):
        """
        Initialize :class:`PowerOddWideAngle`.

        Parameters
        ----------
        d : float, default=1
            Distance at the effective redshift. Use :math:`1` if already included in window functions.

        wa_orders : int, list
            Wide-angle expansion orders.
            So far only order 1 is supported.

        los : string
            Choice of line-of-sight, either:

            - 'firstpoint': the separation vector starts at the end of the line-of-sight
            - 'endpoint': the separation vector ends at the end of the line-of-sight.

        sum_wa : bool, default=True
            Whether to perform summation over wide-angle orders.
            Must be ``False`` if coupling to the window function is to be accounted for.
        """
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
        """x-coordinates are k-wavenumbers."""
        return self.x

    def setup(self, k, projsin, projsout=None):
        r"""
        Set up transform, i.e. compute matrix:

        .. math::

            M_{\ell\ell^{\prime}}^{(n,n^{\prime})}(k) =
            \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right) d} \delta_{\ell,\ell - 1} \delta_{n^{\prime},0} \left[ - \frac{\ell - 1}{k} + \partial_{k} \right]
            - \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right) d} \delta_{\ell,\ell + 1} \delta_{n^{\prime},0} \left[ \frac{\ell + 2}{k} + k \partial_{k} \right]

        if :math:`\ell` is odd and :math:`n = 1`, else:

        .. math::

            M_{\ell\ell^{\prime}}^{(0,n^{\prime})}(k) = \delta_{\ell,ell^{\prime}} \delta_{n^{\prime},0}

        with :math:`\ell` multipole order corresponding to ``projout.proj`` and :math:`\ell^{\prime}` to ``projin.proj``,
        :math:`n` wide angle order corresponding to ``projout.wa_order`` and :math:`n^{\prime}` to ``projin.wa_order``.
        If :attr:`sum_wa` is ``True``, or output ``projout.wa_order`` is ``None``, sum over :math:`n` (only if no window convolution is accounted for).
        Derivatives :math:`\partial_{k}` are computed with finite differences, see arXiv:2106.06324 eq. 3.3.

        Parameters
        ----------
        k : array
            Input (and ouput) wavenumbers.

        projsin : list, ProjectionNameCollection
            Input projections.

        projsout : list, ProjectionNameCollection, default=None
            Output projections. Defaults to ``propose_out(projsin)[-1]``.
        """
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
            if projout.proj % 2 == 0: # even pole :math:`\ell`
                for projin in self.projsin:
                    if projin.proj == projout.proj and (sum_wa or projin.wa_order == projout.wa_order):
                        tmp = eye
                    else:
                        tmp = 0.*eye
                    line.append(tmp)
            else:
                line = [0.*eye for projin in self.projsin]
                if sum_wa:
                    wa_orders = self.wa_orders # sum over :math:`n`
                else:
                    wa_orders = [projout.wa_order] # projout.wa_order is 1
                for wa_order in wa_orders:
                    ells,coeffs = odd_wide_angle_coefficients(projout.proj,wa_order=wa_order,los=self.los)
                    for iprojin,projin in enumerate(self.projsin):
                        if projin.wa_order == 0 and projin.proj in ells:
                            # \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right) d} (if projin.proj == projout.proj - 1)
                            # or - \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right) d} (if projin.proj == projout.proj + 1)
                            coeff = coeffs[ells.index(projin.proj)]/self.d
                            if projin.proj == projout.proj + 1:
                                coeff_spherical_bessel = projin.proj + 1
                            else:
                                coeff_spherical_bessel = -projin.proj
                            # K 'diag' terms arXiv:2106.06324 eq. 3.3, 3.4 and 3.5
                            # tmp is - \frac{\ell \left(\ell - 1\right)}{2 \ell \left(2 \ell - 1\right) d} \frac{\ell - 1}{k} (if projin.proj == projout.proj - 1)
                            # or \frac{\left(\ell + 1\right) \left(\ell + 2\right)}{2 \ell \left(2 \ell + 3\right) d} \frac{\ell + 2}{k} (if projin.proj == projout.proj + 1)
                            tmp = np.diag(coeff_spherical_bessel * coeff / self.k)
                            deltak = 2. * np.diff(self.k)
                            # derivative :math:`\partial_{k}`
                            tmp += np.diag(coeff / deltak, k=1) - np.diag(coeff / deltak, k=-1)

                            # taking care of corners
                            tmp[0,0] -= 2.*coeff / deltak[0]
                            tmp[0,1] = 2.*coeff / deltak[0]
                            tmp[-1,-1] += 2.*coeff / deltak[-1]
                            tmp[-1,-2] = -2.*coeff / deltak[-1]
                            line[iprojin] += tmp
            matrix.append(line)
        self.matrix = np.bmat(matrix).A # (out,in)

    def propose_out(self, projsin):
        """Propose input and output projection names given proposed input projection names ``projsin``."""
        projsin = ProjectionNameCollection(projsin).select(**self.basis.as_dict(drop_none=True)) # restrict to power spectrum multipoles
        #projsin = ProjectionNameCollection([proj.copy(wa_order=proj.wa_order or 0) for proj in projsin if proj.wa_order is None or proj.wa_order % 2 == 0])
        projsin = ProjectionNameCollection([proj for proj in projsin if proj.wa_order is not None])
        ellsin = [proj.proj for proj in projsin if proj.wa_order is not None and proj.wa_order % 2 == 0] # only use input wa_order = 0 multipoles

        projsout = projsin.copy()
        for wa_order in self.wa_orders:
            for ellout in range(1,max(ellsin),2):
                if all(ell in ellsin for ell in odd_wide_angle_coefficients(ellout,wa_order=wa_order)[0]): # check input multipoles are provided
                    projsout.set(projsin[0].copy(wa_order=wa_order,proj=ellout))

        return projsin, projsout
