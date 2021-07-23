import logging

import numpy as np

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib.data_vector import ProjectionNameCollection

from .base import BaseMatrix


class BaseBinning(BaseMatrix):

    logger = logging.getLogger('BaseBinning')

    """Base binning scheme: evaluation right at the data points."""

    def setup(self, data, projs=None):
        self.data = data
        self.projsin = self.projsout = ProjectionNameCollection(projs) if projs is not None else data.get_projs()
        self.xin = self.xout = [self.data.get_x(proj=proj) for proj in self.projsout]
        sout = sin = sum([len(x) for x in self.xin])
        self.matrix = np.eye(sout,sin,dtype='f8')

    def compute(self, array):
        return self.matrix.dot(array)

    def propose_out(self, projsin):
        return self.projsin,self.projsin


class InterpBinning(BaseBinning):

    """Interpolate on the data points."""
    logger = logging.getLogger('InterpBinning')
    regularin = True
    regularout = False

    def __init__(self, xin=None):
        self.xin = np.asarray(xin)

    def setup(self, data, projs=None):
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


class ModeAverageBinning(BaseBinning):
    # one way is to get full data vector here, then average given nmodes
    # in case of pk, better use mesh definition?
    pass


class IntegBinning(BaseBinning):
    # integrate over edge span
    pass


class MeshBinning(BaseBinning):
    # count modes on mesh
    pass
