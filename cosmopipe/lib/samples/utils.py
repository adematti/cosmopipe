from cosmopipe.lib import utils


def nsigmas_to_quantiles_1d(nsigmas):
    from scipy import stats
    return stats.norm.cdf(nsigmas,loc=0,scale=1) - stats.norm.cdf(-nsigmas,loc=0,scale=1)


def nsigmas_to_quantiles_1d_sym(nsigmas):
    total = nsigmas_to_quantiles_1d(nsigmas)
    out = (1.-total)/2.
    return out,1.-out


def nsigmas_to_deltachi2(nsigmas, ndof=1):
    from scipy import stats
    quantile = nsigmas_to_quantiles_1d(nsigmas)
    return stats.chi2.ppf(quantile,ndof)


def metrics_to_latex(name):
    toret = utils.txt_to_latex(name)
    for full,symbol in [('loglkl','L'),('logposterior','\\mathcal{L}'),('logprior','p')]:
        toret = toret.replace(full,symbol)
    return toret
