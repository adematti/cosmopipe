import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

from cosmopipe.lib import plotting, utils
from cosmopipe.lib.parameter import ParamBlock, Parameter

from .mesh import Mesh
from .samples import Samples
from .profiles import Profiles
from .utils import *


def lighten_color(color, amount=0.5):
    """Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)

    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    lum = 1 - amount * (1 - c[1]) if amount > 0 else - amount * c[1]
    return colorsys.hls_to_rgb(c[0], lum, c[2])


def get_color_sequence(colors, ncolors=1, lighten=-0.5):
    toret = []
    if isinstance(colors,list):
        toret += colors
    else:
        toret += [colors]
    for i in range(ncolors-len(toret)): toret.append(lighten_color(toret[-1],amount=lighten))
    return toret


def plot_samples_1d(ax, samples, parameter, normalise='max', method='gaussian_kde', bins=60, **kwargs):
    ax.set_ylim(bottom=0)
    samples.add_default_weight()
    if not isinstance(samples,Mesh):
        mesh = samples.to_mesh(parameters=[parameter],method=method,bins=bins)
    else:
        mesh = samples
    x,pdf = mesh(parameters=[parameter])
    if normalise == 'max': pdf /= pdf.max()
    if method == 'histo':
        ax.hist(x,bins=mesh.edges[0],weights=pdf,density=False,**kwargs)
    else:
        ax.plot(x,pdf,**kwargs)


def plot_samples_2d(ax, samples, parameters, sigmas=2, method='gaussian_kde', scatter=False, bins=60, colors=None, fill=True, lighten=-0.5, **kwargs):
    if scatter:
        ax.scatter(samples[parameters[0]],samples[parameters[1]],**kwargs)
        if scatter == 'only': return
    if not isinstance(samples,Mesh):
        mesh = samples.to_mesh(parameters=parameters,method=method,bins=bins)
    else:
        mesh = samples
    levels = mesh.get_sigmas(sigmas)
    (x,y),pdf = mesh(parameters=parameters)
    if colors is not None:
        colors = get_color_sequence(colors,ncolors=len(levels),lighten=lighten)
    if fill:
        ax.contourf(x,y,pdf.T,levels=levels[::-1]+[pdf.max()+1.],colors=colors,**kwargs)
    else:
        ax.contour(x,y,pdf.T,levels=levels[::-1],colors=colors,**kwargs)


def plot_normal_1d(ax, mean=0., covariance=1., lim=None, normalize='max', **kwargs):

    if lim is None: lim = ax.get_xlim()
    x = np.linspace(*lim,num=1000)
    y = np.exp(-(x-mean)**2/2./covariance)
    if normalize != 'max': y *= 1./(covariance*np.sqrt(2.*np.pi))
    ax.plot(x,y,**kwargs)


def plot_contour_2d(ax, contours, sigmas=2, colors=None, fill=False, lighten=-0.5, **kwargs):

    if not isinstance(contours,(tuple,list)):
        contours = [contours]
    nsigmas = len(contours)
    if colors is not None:
        colors = get_color_sequence(colors,ncolors=nsigmas,lighten=lighten)
    else:
        colors = [None]*nsigmas

    for isigma,color in zip(reversed(range(nsigmas)),colors):
        x,y = (np.concatenate([xy,xy[:1]],axis=0) for xy in contours[isigma])
        if fill: ax.fill(x,y,color=color,**kwargs)
        else: ax.plot(x,y,color=color,**kwargs)


def plot_normal_2d(ax, mean=None, covariance=None, sigmas=2, **kwargs):

    if np.isscalar(sigmas): sigmas = 1 + np.arange(sigmas)

    radii = np.sqrt(nsigmas_to_deltachi2(sigmas,ndof=2))
    t = np.linspace(0.,2.*np.pi,1000,endpoint=False)
    ct = np.cos(t)
    st = np.sin(t)
    covariance = np.array(covariance)
    if covariance.size == 3:
        sigx2,sigy2,sigxy = covariance
    else:
        sigx2,sigy2,sigxy = [covariance[i,j] for i,j in [(0,0),(1,1),(0,1)]]

    contours = []
    for radius in radii:
        a = radius * np.sqrt(0.5 * (sigx2 + sigy2) + np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
        b = radius * np.sqrt(0.5 * (sigx2 + sigy2) - np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
        th = 0.5 * np.arctan2(2. * sigxy, sigx2 - sigy2)
        x = mean[0] + a * ct * np.cos(th) - b * st * np.sin(th)
        y = mean[1] + a * ct * np.sin(th) + b * st * np.cos(th)
        contours.append((x,y))

    plot_contour_2d(ax, contours, **kwargs)


def univariate_gelman_rubin(chains):
    """
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
    dim 0: nchains
    dim 1: nsteps
    """
    nchains = len(chains)
    mean = np.asarray([np.mean(chain,axis=0) for chain in chains])
    variance = np.asarray([np.var(chain,ddof=1,axis=0) for chain in chains])
    nsteps = np.asarray([len(chain) for chain in chains])
    Wn1 = np.mean(variance)
    Wn = np.mean((nsteps-1.)/nsteps*variance)
    B = np.var(mean,ddof=1)
    V = Wn + (nchains+1.)/nchains*B
    return V/Wn1


def multivariate_gelman_rubin(chains):
    """
    http://www.stat.columbia.edu/~gelman/research/published/brooksgelman2.pdf
    dim 0: nchains
    dim 1: nsteps
    dim 2: ndim
    """
    nchains = len(chains)
    mean = np.asarray([np.mean(chain,axis=0) for chain in chains])
    variance = np.asarray([np.cov(chain.T,ddof=1) for chain in chains])
    nsteps = np.asarray([len(chain) for chain in chains])
    Wn1 = np.mean(variance,axis=0)
    Wn = np.mean(((nsteps-1.)/nsteps)[:,None,None]*variance,axis=0)
    B = np.cov(mean.T,ddof=1)
    V = Wn + (nchains+1.)/nchains*B
    invWn1 = np.linalg.inv(Wn1)
    assert np.absolute(Wn1.dot(invWn1)-np.eye(Wn1.shape[0])).max() < 1e-5
    eigen = np.linalg.eigvalsh(invWn1.dot(V))
    return eigen.max()


def autocorrelation_1d(x):
    """Estimate the normalized autocorrelation function of a 1-D series

    Args:
        x: The series as a 1-D numpy array.

    Returns:
        array: The autocorrelation function of the time series.

    Taken from: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
    """
    from numpy import fft
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")

    def next_pow_two(n):
        """Returns the next power of two greater than or equal to `n`"""
        i = 1
        while i < n:
            i = i << 1
        return i
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = fft.fft(x - np.mean(x), n=2*n)
    acf = fft.ifft(f * np.conjugate(f))[:len(x)].real

    acf /= acf[0]
    return acf


def integrated_autocorrelation_time(x, c=5, tol=50, quiet=False):
    """Estimate the integrated autocorrelation time of a time series.

    This estimate uses the iterative procedure described on page 16 of
    `Sokal's notes <http://www.stat.unc.edu/faculty/cji/Sokal.pdf>`_ to
    determine a reasonable window size.

    Args:
        x: The time series. If multidimensional, set the time axis using the
            ``axis`` keyword argument and the function will be computed for
            every other axis.
        c (Optional[float]): The step size for the window search. (default:
            ``5``)
        tol (Optional[float]): The minimum number of autocorrelation times
            needed to trust the estimate. (default: ``50``)
        quiet (Optional[bool]): This argument controls the behavior when the
            chain is too short. If ``True``, give a warning instead of raising
            an :class:`AutocorrError`. (default: ``False``)

    Returns:
        float or array: An estimate of the integrated autocorrelation time of
            the time series ``x`` computed along the axis ``axis``.

    Raises
        ValueError: If the autocorrelation time can't be reliably estimated
            from the chain and ``quiet`` is ``False``. This normally means
            that the chain is too short.

    Taken from: https://github.com/dfm/emcee/blob/master/emcee/autocorr.py
    """
    x = np.atleast_1d(x)
    if len(x.shape) == 1:
        x = x[:, np.newaxis, np.newaxis]
    if len(x.shape) == 2:
        x = x[:, :, np.newaxis]
    if len(x.shape) != 3:
        raise ValueError("Invalid dimensions")

    n_t, n_w, n_d = x.shape
    tau_est = np.empty(n_d)
    windows = np.empty(n_d, dtype=int)

    def auto_window(taus, c):
        m = np.arange(len(taus)) < c * taus
        if np.any(m):
            return np.argmin(m)
        return len(taus) - 1

    # Loop over parameters
    for d in range(n_d):
        f = np.zeros(n_t)
        for k in range(n_w):
            f += autocorrelation_1d(x[:, k, d])
        f /= n_w
        taus = 2.0*np.cumsum(f)-1.0
        windows[d] = auto_window(taus, c)
        tau_est[d] = taus[windows[d]]

    # Check convergence
    flag = tol * tau_est > n_t

    # Warn or raise in the case of non-convergence
    if np.any(flag):
        msg = (
            "The chain is shorter than {:d} times the integrated "
            "autocorrelation time for {:d} parameter(s). Use this estimate "
            "with caution and run a longer chain!\n"
        ).format(tol, np.sum(flag))
        msg += "N/{:d} = {:.0f};\ntau: {}".format(tol,n_t*1./tol,tau_est)
        if not quiet: raise ValueError(msg)
        logger.warning(msg)

    return tau_est


def make_list(obj, length=1):
    if isinstance(obj,tuple):
        return list(obj)
    if not isinstance(obj,(list,np.ndarray)):
        return [obj for i in range(length)]
    return obj


class ListPlotStyle(plotting.BasePlotStyle):

    def get_color(self, label, labels=None):
        if labels is None:
            labels = self.labels
            index = self.labels.index(label)
        else:
            index = label
            label = labels[index]
        if self.colors is None:
            color = 'C{:d}'.format(index)
        elif isinstance(self.colors,str):
            color = self.colors
        elif isinstance(self.colors,dict):
            color = self.colors[label]
        else:
            color = self.colors[index]
        return color

    def get_list(self, name, value=None, default=None):
        if value is not None:
            value = make_list(value,length=len(default) if default is not None else 1)
            return value
        value = getattr(self,name,None)
        if value is None:
            return default
        return make_list(value,length=len(default) if default is not None else 1)

    @staticmethod
    def _get_default_parameters(parameters, samples):
        if parameters is None:
            parameters = samples.parameters.select(fixed=False)
        return parameters

    @staticmethod
    def get_parameters(parameters, chains=None):
        isscalar = not isinstance(parameters,(tuple,list,ParamBlock))
        if isscalar:
            parameters = [parameters]
        toret = ParamBlock(parameters)
        if chains is None:
            if isscalar:
                return toret[0]
            return toret
        elif not isinstance(chains,(tuple,list)):
            chains = [chains]
        for iparam,param in enumerate(toret):
            if isinstance(parameters[iparam],Parameter):
                continue
            for ichain,chain in enumerate(chains):
                if param.name in chain.parameters:
                    toret[iparam] = chain.parameters[param.name]
                    break
        if isscalar:
            return toret[0]
        return toret

    def _get_default_truths(self, truths, parameters):
        isscalar = not isinstance(parameters,(tuple,list,ParamBlock))
        if isscalar:
            truths = [truths]
            parameters = [parameters]
        truths = self.get_list('truths',truths,[None]*len(parameters))
        toret = [param.value if truth == 'value' else truth for truth,param in zip(truths,parameters)]
        if isscalar:
            return toret[0]
        return toret


class SamplesPlotStyle(ListPlotStyle):

    def __init__(self, style=None, **kwargs):
        super(SamplesPlotStyle,self).__init__(style=style)
        self.figsize = None
        self.title = None
        self.title_interval = 'quantile'
        self.kwplt_1d = {}
        self.kwplt_2d = {'alpha':0.8}
        self.kwplt_truth = {'linestyle':'--','linewidth':1,'color':'k'}
        self.majorlocator = MaxNLocator(nbins=3,min_n_ticks=2,prune='both')
        self.minorlocator = AutoMinorLocator(2)
        self.kwticklabel = {'scilimits':(-3,3)}
        self.titlesize = 18
        self.labelsize = 18
        self.ticksize = 14
        self.chains = None
        self.labels = None
        self.fills = None
        self.truths = None
        self.colors = None
        self.kwlegend = {'ncol':1,'fontsize':self.labelsize,'framealpha':1,'frameon':True}
        self.filename = None
        self.update(**kwargs)

    def plot_1d(self, chains=None, parameter=None, labels=None, truth=None, ax=None, filename=None):
        tosave = ax is None
        if tosave: ax = plt.gca()
        chains = self.get_list('chains',chains)
        parameter = self.get_parameters(parameter,chains=chains)
        truth = self._get_default_truths(truth,parameter)
        labels = self.get_list('labels',labels,[getattr(chain,'name',None) for chain in chains])
        add_legend = any(label is not None for label in labels)
        for ichain,(chain,label) in enumerate(zip(chains,labels)):
            plot_samples_1d(ax,chain,parameter,label=label,color=self.get_color(ichain,labels=labels),**self.kwplt_1d)
        if truth is not None:
            ax.axvline(x=truth,ymin=0.,ymax=1.,**self.kwplt_truth)
        ax.set_xlabel(parameter.get_label(),fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        if add_legend: ax.legend(**self.kwlegend)
        if self.title_interval == 'quantile' and len(chains) == 1:
            q = nsigmas_to_quantiles_1d_sym(1)
            qm, ql, qh = chains[0].quantile(parameter,q=(0.5,)+q)
            title = '{} = ${{{}}}_{{-{}}}^{{+{}}}$'.format(parameter.get_label(),*utils.round_measurement(qm,qm-ql,qh-qm,sigfigs=2))
            ax.set_title(title,fontsize=self.labelsize)
        if tosave:
            filename = filename or self.filename
            if filename: self.savefig(filename)
        return ax

    def plot_2d(self, chains=None, parameters=None, labels=None, truths=None, ax=None, filename=None):
        tosave = ax is None
        if tosave: ax = plt.gca()
        chains = self.get_list('chains',chains)
        parameters = self.get_parameters(parameters,chains=chains)
        truths = self._get_default_truths(truths,parameters)
        labels = self.get_list('labels',labels,[getattr(chain,'name',None) for chain in chains])
        fills = self.get_list('fills',default=[True]*len(chains))
        handles = []
        add_legend = any(label is not None for label in labels)
        for ichain,(chain,label,fill) in enumerate(zip(chains,labels,fills)):
            plot_samples_2d(ax,chain,parameters,colors=self.get_color(ichain,labels=labels),fill=fill,**self.kwplt_2d)
            if label is not None: handles.append(patches.Patch(color=self.get_color(ichain,labels=labels),label=label,alpha=1))
        if truths is not None:
            if truths[0] is not None: ax.axvline(x=truths[0],ymin=0.,ymax=1.,**self.kwplt_truth)
            if truths[1] is not None: ax.axhline(y=truths[1],xmin=0.,xmax=1.,**self.kwplt_truth)
        ax.set_xlabel(parameters[0].get_label(),fontsize=self.labelsize)
        ax.set_ylabel(parameters[1].get_label(),fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        if add_legend: ax.legend(**{**{'handles':handles},**self.kwlegend})
        if tosave:
            filename = filename or self.filename
            if filename: self.savefig(filename)
        return ax

    def plot_corner(self, chains=None, parameters=None, labels=None, truths=None, filename=None):
        chains = self.get_list('chains',chains)
        parameters = self._get_default_parameters(parameters,chains[0])
        parameters = self.get_parameters(parameters,chains=chains)
        truths = self._get_default_truths(truths,parameters)
        labels = self.get_list('labels',labels,[getattr(chain,'name',None) for chain in chains])
        handles = []
        add_legend = any(label is not None for label in labels)
        ncols = nrows = len(parameters)
        figsize = self.figsize or (3.3*ncols,2.7*nrows)
        fig = plt.figure(figsize=figsize)
        if self.title: fig.suptitle(title,fontsize=self.titlesize)
        gs = gridspec.GridSpec(nrows,ncols,wspace=0.1,hspace=0.1)
        dax = {}
        xlims,xticks = [],{True:[],False:[]}
        for iparam1,param1 in enumerate(parameters):
            ax = dax[param1.name] = plt.subplot(gs[iparam1,iparam1])
            self.plot_1d(chains,param1,truth=truths[iparam1],ax=ax)
            leg = ax.get_legend()
            if leg is not None: leg.remove()
            for ichain1,chain1 in enumerate(chains):
                if labels[ichain1] is not None: handles.append(patches.Patch(color=self.get_color(ichain1,labels=labels),label=labels[ichain1],alpha=1))
            if iparam1 < nrows-1: ax.get_xaxis().set_visible(False)
            #else: ax.set_xlabel(param1.get_label(),fontsize=self.labelsize)
            ax.get_yaxis().set_visible(False)
            ax.tick_params(labelsize=self.ticksize)
            xlims.append(ax.get_xlim())
            ax.xaxis.set_major_locator(self.majorlocator)
            ax.xaxis.set_minor_locator(self.minorlocator)
            for minor in xticks: xticks[minor].append(ax.get_xticks(minor=minor))
            ax.ticklabel_format(**self.kwticklabel)
            if add_legend and iparam1 == 0: ax.legend(**{**{'loc':'upper left','handles':handles,'bbox_to_anchor':(1.04,1.)},**self.kwlegend})

        for iparam1,param1 in enumerate(parameters):
            for iparam2,param2 in enumerate(parameters):
                if nrows-1-iparam2 >= ncols-1-iparam1: continue
                ax = dax[param1.name,param2.name] = plt.subplot(gs[iparam2,iparam1])
                self.plot_2d(chains,(param1,param2),truths=(truths[iparam1],truths[iparam2]),ax=ax)
                leg = ax.get_legend()
                if leg is not None: leg.remove()
                if iparam1>0: ax.get_yaxis().set_visible(False)
                #else: ax.set_ylabel(param2.get_label(),fontsize=self.labelsize)
                if nrows-1-iparam2>0: ax.get_xaxis().set_visible(False)
                #else: ax.set_xlabel(param1.get_label(),fontsize=self.labelsize)
                ax.tick_params(labelsize=self.ticksize)
                for minor in xticks:
                    ax.set_xticks(xticks[minor][iparam1],minor=minor)
                    ax.set_yticks(xticks[minor][iparam2],minor=minor)
                ax.ticklabel_format(**self.kwticklabel)
                ax.set_xlim(xlims[iparam1])
                ax.set_ylim(xlims[iparam2])

        filename = filename or self.filename
        if filename: self.savefig(filename)
        return fig,dax

    def plot_chain(self, chain, parameters=None, filename=None):
        parameters = self._get_default_parameters(parameters,chain)
        parameters = self.get_parameters(parameters,chains=chain)
        nparams = len(parameters)
        steps = 1 + np.arange(len(chain))
        figsize = self.figsize or (8,1.5*nparams)
        fig,lax = plt.subplots(nparams,sharex=True,sharey=False,figsize=figsize,squeeze=False)
        lax = lax.flatten()

        for ax,param in zip(lax,parameters):
            ax.grid(True)
            ax.tick_params(labelsize=self.labelsize)
            ax.set_ylabel(param.get_label(),fontsize=self.labelsize)
            ax.plot(steps,chain[param],color='k')

        lax[-1].set_xlabel('step',fontsize=self.labelsize)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return fig,lax

    def plot_gelman_rubin(self, chains=None, parameters=None, multivariate=False, threshold=1.1, slices=None, ax=None, filename=None):
        chains = self.get_list('chains',chains)
        parameters = self._get_default_parameters(parameters,chains[0])
        parameters = self.get_parameters(parameters,chains=chains)
        if slices is None:
            nsteps = np.amin([len(chain) for chain in chains])
            slices = np.arange(100,nsteps,500)
        gr_multi = []
        gr = {param.name:[] for param in parameters}
        for end in slices:
            if multivariate: gr_multi.append(multivariate_gelman_rubin([np.asarray([chain[param.name][:end] for param in parameters]).T for chain in chains]))
            for param in gr: gr[param].append(univariate_gelman_rubin([chain[param][:end] for chain in chains]))
        for param in gr: gr[param] = np.asarray(gr[param])

        if ax is None: ax = plt.gca()
        ax.grid(True)
        ax.tick_params(labelsize=self.ticksize)
        ax.set_xlabel('step',fontsize=self.labelsize)
        ax.set_ylabel('$\\hat{R}$',fontsize=self.labelsize)

        if multivariate: ax.plot(slices,gr_multi,label='multi',linestyle='-',linewidth=1,color='k')
        for param in parameters:
            ax.plot(slices,gr[param.name],label=param.get_label(),linestyle='--',linewidth=1)
        if threshold is not None: ax.axhline(y=threshold,xmin=0.,xmax=1.,linestyle='--',linewidth=1,color='k')
        ax.legend(**self.kwlegend)

        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax

    def plot_autocorrelation_time(self, chains=None, parameters=None, threshold=50, slices=None, ax=None, filename=None):
        chains = self.get_list('chains',chains)
        parameters = self._get_default_parameters(parameters,chains[0])
        parameters = self.get_parameters(parameters,chains=chains)
        if slices is None:
            nsteps = np.amin([len(chain) for chain in chains])
            slices = np.arange(100,nsteps,500)

        autocorr = {param.name:[] for param in parameters}
        for end in slices:
            for param in autocorr:
                autocorr[param].append(integrated_autocorrelation_time(np.asarray([chain[param][:end] for chain in chains]).T,tol=0))
        for param in autocorr: autocorr[param] = np.asarray(autocorr[param])

        if ax is None: ax = plt.gca()
        ax.grid(True)
        ax.tick_params(labelsize=self.ticksize)
        ax.set_xlabel('step $N$',fontsize=self.labelsize)
        ax.set_ylabel('$\\tau$',fontsize=self.labelsize)

        for param in parameters:
            ax.plot(slices,autocorr[param.name],label=param.get_label(),linestyle='--',linewidth=1)
        if threshold is not None:
            ax.plot(slices,slices*1./threshold,label='$N/{:d}$'.format(threshold),linestyle='--',linewidth=1,color='k')
        ax.legend(**self.kwlegend)

        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax


class ProfilesPlotStyle(ListPlotStyle):

    def __init__(self, style=None, **kwargs):

        super(ProfilesPlotStyle,self).__init__(style=style)
        self.figsize = None
        self.title = None
        self.kwplt_truth = {'linestyle':'--','linewidth':2,'color':'k'}
        self.kwplt_scatter = {'marker':'o','markersize':4,'markeredgecolor':'none','elinewidth':1}
        self.kwplt_bands = {'facecolor':'k','alpha':0.1,'linewidth':0,'linestyle':None}
        self.kwplt_means = {'color':'k','elinewidth':1,'marker':'o','markersize':4}
        self.titlesize = 18
        self.labelsize = 18
        self.ticksize = 14
        self.labels = None
        self.fills = None
        self.truths = None
        self.colors = None
        self.ybands = None
        self.ylimits = None
        self.filename = None
        self.means = False
        self.errors = 'parabolic_errors'
        self.kwlegend = {'loc':'upper left','fontsize':self.labelsize,'framealpha':1,'frameon':True,'bbox_to_anchor':(0.,1.3)}
        self.kwplt_1d = {'method':'histo','histtype':'step','bins':20}
        self.kwplt_2d = {'bins':10,'scatter':'only','alpha':0.8}
        self.kstest = 'norm'
        self.title_interval = True
        self.majorlocator = MaxNLocator(nbins=3,min_n_ticks=2,prune='both')
        self.minorlocator = AutoMinorLocator(2)
        self.kwticklabel = {'scilimits':(-3,3)}
        self.update(**kwargs)


    def plot_aligned(self, profiles=None, parameter=None, ids=None, labels=None, truth=None, yband=None, ax=None, filename=None):

        tosave = ax is None
        if tosave: ax = plt.gca()
        profiles = self.get_list('profiles',profiles)
        parameter = self.get_parameters(parameter,chains=profiles)
        truth = self._get_default_truths(truth,parameter)
        if ids is None: ids = [None] * len(profiles)
        maxpoints = max(map(len,profiles))
        labels = self.get_list('labels',labels,[None]*maxpoints)
        add_legend = any(label is not None for label in labels)

        xmain = np.arange(len(profiles))
        xaux = np.linspace(-0.15,0.15,maxpoints)
        for iprof,prof in enumerate(profiles):
            if parameter not in prof.bestfit: continue
            ibest = prof.argmin()
            for ipoint,point in enumerate(prof.bestfit[parameter]):
                yerr = None
                if self.errors == 'parabolic_errors':
                    yerr = prof.get(self.errors)[parameter][ibest]
                elif ipoint == ibest:
                    yerr = prof.get(self.errors)[parameter]
                label = labels[ipoint] if iprof == 0 else None
                ax.errorbar(xmain[iprof]+xaux[ipoint],point,yerr=yerr,color=self.get_color(ipoint,labels=labels),
                            label=label,linestyle='none',**self.kwplt_scatter)
            if self.means:
                ax.errorbar(xmain[iprof],prof.bestfit[parameter].mean(),yerr=prof.bestfit[parameter].std(ddof=1),linestyle='none',**self.kwplt_means)
        if truth is not None:
            ax.axhline(y=truth,xmin=0.,xmax=1.,**self.kwplt_truth)
        if yband is not None:
            if yband[-1] == 'abs': low,up = yband[0],yband[1]
            else: low,up = yband*(1-yband[0]),yband*(1+yband[1])
            ax.axhspan(low,up,**self.kwplt_bands)

        ax.set_xticks(xmain)
        ax.set_xticklabels(ids,rotation=40,ha='right',fontsize=self.labelsize)
        ax.grid(True,axis='y')
        ax.set_ylabel(parameter.get_label(),fontsize=self.labelsize)
        ax.tick_params(labelsize=self.ticksize)
        if add_legend: ax.legend(**{**{'ncol':maxpoints},**self.kwlegend})
        if tosave:
            filename = filename or self.filename
            if filename: self.savefig(filename)
        return ax

    def plot_aligned_stacked(self, profiles=None, parameters=None, ids=None, labels=None, truths=None, ybands=None, ylimits=None, filename=None):

        profiles = self.get_list('profiles',profiles)
        parameters = self._get_default_parameters(parameters,profiles[0])
        truths = self._get_default_truths(truths,parameters)
        ybands = self.get_list('ybands',ybands,[None]*len(parameters))
        ylimits = self.get_list('ylimits',ylimits,[None]*len(parameters))
        maxpoints = max(map(len,profiles))

        nrows = len(parameters)
        ncols = len(profiles) if len(profiles) > 1 else maxpoints
        figsize = self.figsize or (ncols,3.*nrows)
        fig = plt.figure(figsize=figsize)
        if self.title: fig.suptitle(title,fontsize=self.titlesize)
        gs = gridspec.GridSpec(nrows,1,wspace=0.1,hspace=0.1)

        lax = []
        for iparam1,param1 in enumerate(parameters):
            ax = plt.subplot(gs[iparam1])
            self.plot_aligned(profiles,parameter=param1,ids=ids,labels=labels,truth=truths[iparam1],yband=ybands[iparam1],ax=ax)
            if (iparam1 < nrows-1) or not ids: ax.get_xaxis().set_visible(False)
            ax.set_ylim(ylimits[iparam1])
            if iparam1 != 0:
                leg = ax.get_legend()
                if leg is not None: leg.remove()
            lax.append(ax)

        filename = filename or self.filename
        if filename: self.savefig(filename)
        return lax

    def _profiles_to_samples(self, profiles, parameters, select='best', residual='parabolic_errors', truths=None):
        profiles = self.get_list('profiles',profiles)
        if isinstance(profiles[0],Samples):
            return profiles

        def to_samples(profiles, parameters):
            parameters = self._get_default_parameters(parameters,profiles[0])
            toret = Profiles.to_samples(profiles,parameters=parameters,name='bestfit',select=select)
            if residual:
                for iparam,param in enumerate(parameters):
                    if truths[iparam] is not None:
                        toret[param] -= truths[iparam]
                    else:
                        toret[param] -= toret.mean(param)
                errors = Profiles.to_samples(profiles,name=residual,select=select)
                for param in parameters:
                    if residual == 'parabolic_errors':
                        toret[param] /= errors[param]
                    else:
                        mask = toret[param] > 0
                        toret[param][mask] /= errors[param][mask,0]
                        toret[param][~mask] /= errors[param][~mask,0]
            return toret

        if isinstance(profiles[0],Profiles):
            return [to_samples(profiles,parameters)]
        return [to_samples(prof,parameters) for prof in profiles]

    @staticmethod
    def _get_label(parameter, residual='parabolic_errors'):
        if residual:
            if parameter.latex is not None:
                return '$\Delta {0}/\sigma({0})$'.format(parameter.latex)
            return '$\Delta$ {0}/sigma({0})'.format(parameter.name)
        return parameter.get_label()

    def plot_1d(self, profiles=None, parameter=None, select='best', residual='parabolic_errors', truth=None, filename=None, **kwargs):
        truths = [self._get_default_truths(truth,parameter)]
        chains = self._profiles_to_samples(profiles=profiles,parameters=[parameter],select=select,residual=residual,truths=truths)
        parameter = self.get_parameters(parameter,chains=chains)
        ax = SamplesPlotStyle.plot_1d(self,chains,parameter=parameter,**kwargs)
        ax.set_xlabel(self._get_label(parameter,residual=residual))
        if self.kstest and len(chains) == 1:
            from scipy import stats
            d,p = stats.kstest(chains[0][parameter],self.kstest,alternative='two-sided')
            text = '$(D_{{n}},p) = ({:.3f},{:.3f})$'.format(d,p)
            ax.text(0,1.2,text,horizontalalignment='left',verticalalignment='top',transform=ax.transAxes,color='k',fontsize=self.labelsize)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax

    def plot_2d(self, profiles=None, parameters=None, select='best', residual='parabolic_errors', truths=None, filename=None, **kwargs):
        truths = self._get_default_truths(truths,parameters)
        chains = self._profiles_to_samples(profiles=profiles,parameters=parameters,select=select,residual=residual,truths=truths)
        parameters = self.get_parameters(parameters,chains=chains)
        ax = SamplesPlotStyle.plot_2d(self,chains,parameters=parameters,**kwargs)
        ax.set_xlabel(self._get_label(parameters[0],residual=residual))
        ax.set_ylabel(self._get_label(parameters[1],residual=residual))
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax

    def plot_corner(self, profiles=None, parameters=None, select='best', residual='parabolic_errors', truths=None, **kwargs):
        truths = self._get_default_truths(truths,parameters)
        chains = self._profiles_to_samples(profiles=profiles,parameters=parameters,select=select,residual=residual,truths=truths)
        return SamplesPlotStyle.plot_corner(self,chains,parameters=parameters,**kwargs)
