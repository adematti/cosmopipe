"""Utilities for confidence level conversions."""

from scipy import stats

from cosmopipe.lib import utils


def nsigmas_to_quantiles_1d(nsigmas):
    r"""
    Turn number of Gaussian sigmas ``nsigmas`` into quantiles,
    e.g. :math:`\simeq 0.68` for :math:`1 \sigma`.
    """
    return stats.norm.cdf(nsigmas,loc=0,scale=1) - stats.norm.cdf(-nsigmas,loc=0,scale=1)


def nsigmas_to_quantiles_1d_sym(nsigmas):
    r"""
    Turn number of Gaussian sigmas ``nsigmas`` into lower and upper quantiles,
    e.g. :math:`\simeq 0.16, 0.84` for :math:`1 \sigma`.
    """
    total = nsigmas_to_quantiles_1d(nsigmas)
    out = (1.-total)/2.
    return out,1.-out


def nsigmas_to_deltachi2(nsigmas, ddof=1):
    r"""Turn number of Gaussian sigmas ``nsigmas`` into :math:`\chi^{2}` levels at ``ddof`` degrees of freedom."""
    quantile = nsigmas_to_quantiles_1d(nsigmas)
    return stats.chi2.ppf(quantile,ndof) # inverse of cdf


def metrics_to_latex(name):
    """Turn metrics ``name`` to latex string."""
    toret = utils.txt_to_latex(name)
    for full,symbol in [('loglkl','L'),('logposterior','\\mathcal{L}'),('logprior','p')]:
        toret = toret.replace(full,symbol)
    return toret
