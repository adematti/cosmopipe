"""Classes and functions dedicated to theory models."""

from .base import ProjectionBasis, ProjectionBasisCollection, ModelCollection
from .linear import LinearModel
from .evaluation import ModelEvaluation
from .effect_ap import AnisotropicScaling, IsotropicScaling
from .gaussian_covariance import GaussianCovarianceMatrix
