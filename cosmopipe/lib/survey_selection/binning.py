"""Classes defining final projection onto the data points."""

import logging

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import ProjectionNameCollection

from .base import BaseMatrix


class BaseBinning(BaseMatrix):

    """Class handling base binning scheme, i.e. evaluation right at the data points."""

    def setup(self, data, projs=None):
        """
        Set up projection.

        Parameters
        ----------
        data : DataVector
            Data vector to project onto.

        projs : list, ProjectionNameCollection, default=None
            Projection names.
            If ``None``, defaults to ``data.get_projs()``, i.e. projections within data view.
        """
        self.data = data
        self.projsin = self.projsout = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
        self.xin = self.xout = [self.data.get_x(proj=proj) for proj in self.projsout]
        sout = sin = sum([len(x) for x in self.xin])
        self.matrix = np.eye(sout,sin,dtype='f8')

    def propose_out(self, projsin):
        """Propose input and output projection names: exactly those of data vector."""
        return self.projsin,self.projsin


class InterpBinning(BaseBinning):

    """Class handling model interpolation at the data points."""

    regularin = True
    regularout = False

    def __init__(self, xin):
        """
        Initialize :class:`InterpBinning`.

        Parameters
        ----------
        xin : array
            x-coordinates to evaluate model at (before interpolation).
        """
        self.xin = np.asarray(xin)

    def setup(self, data, projs=None):
        """
        Set up projection.

        Parameters
        ----------
        data : DataVector
            Data vector to project onto.

        projs : list, ProjectionNameCollection, default=None
            Projection names.
            If ``None``, defaults to ``data.get_projs()``, i.e. projections within data view.
        """
        self.data = data
        self.projsin = self.projsout = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
        self.xout = [self.data.get_x(proj=proj) for proj in self.projsout]
        matrix = []
        zeros = [np.zeros((len(xout),len(self.xin)),dtype='f8') for xout in self.xout]
        for iproj,xout in enumerate(self.xout):
            frac = np.interp(xout,self.xin,np.arange(len(self.xin)))
            tmp = zeros[iproj].copy()
            index = np.int32(frac)
            for ii,jj in enumerate(index):
                diff = frac[ii] - jj
                tmp[ii,jj] = 1. - diff
                if diff > 0.:
                    tmp[ii,jj+1] = diff
            line = zeros.copy()
            line[iproj] = tmp
            matrix.append(line)
        self.matrix = np.bmat(matrix).A


#class ModeAverageBinning(BaseBinning):
    # one way is to get full data vector here, then average given nmodes
    # in case of pk, better use mesh definition?


#class IntegBinning(BaseBinning):
    # integrate over edge span


#class MeshBinning(BaseBinning):
    # count modes on mesh
