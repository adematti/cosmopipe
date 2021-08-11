"""Utilities to plot likelihood samples and profiles."""

import numpy as np

from matplotlib import pyplot as plt
from matplotlib import gridspec, patches
from matplotlib.ticker import MaxNLocator, AutoMinorLocator

from cosmopipe.lib import plotting, utils
from cosmopipe.lib.parameter import ParameterCollection, Parameter

from .mesh import Mesh
from .samples import Samples
from .profiles import Profiles
from .utils import nsigmas_to_quantiles_1d_sym, nsigmas_to_deltachi2


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying 1-luminosity by the given amount ``amount``.

    Taken from https://stackoverflow.com/questions/37765197/darken-or-lighten-a-color-in-matplotlib

    >>> lighten_color('g', 0.3)
    >>> lighten_color('#F034A3', 0.6)
    >>> lighten_color((.3,.55,.1), 0.5)

    Parameters
    ----------
    color : string, tuple
         *matplotlib* color string, hex string, or RGB tuple.

    amount : float, default=0.5
        Positive number ligthens color, negative darkens color.

    Returns
    -------
    color : tuple
        RGB tuple.
    """
    import matplotlib.colors as mc
    import colorsys
    try:
        c = mc.cnames[color]
    except KeyError:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    lum = 1 - amount * (1 - c[1]) if amount > 0 else - amount * c[1]
    return colorsys.hls_to_rgb(c[0], lum, c[2])


def get_color_sequence(colors, ncolors=1, lighten=-0.5):
    """
    Return list of colors of varying luminosity.

    Parameters
    ----------
    colors : list, string, tuple
        Single or list of colors to start sequence with.

    ncolors : int, default=1
        Number of colors to return.

    lighten : float, default=-0.5
        Positive number ligthens colors, negative darkens colors.

    Returns
    -------
    colors : list
        List of RGB colors (tuples).
    """
    toret = []
    if isinstance(colors,list):
        toret += colors
    else:
        toret += [colors]
    for i in range(ncolors-len(toret)): toret.append(lighten_color(toret[-1],amount=lighten))
    return toret


def plot_samples_1d(ax, samples, parameter, normalize='max', method='gaussian_kde', bins=60, **kwargs):
    """
    Plot 1D samples for parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where to plot samples.

    samples : Samples
        Samples to plot.

    parameter : Parameter, ParamName
        Parameter to plot samples for.

    normalize : string, default='max'
        How to normalize 1D distribution.
        Use 'max' to set maximum at 1 (most common),
        else distribution will be normalized as probability density (integral of 1).

    method : string
        Method to interpolate samples on mesh, either:

        - 'gaussian_kde': Gaussian kernel density estimation, based on :class:`scipy.stats.gaussian_kde`
        - 'cic' : Cloud-in-Cell assignment
        - 'histo' : simple binning.

    bins : int, default=60
        Number of bins i.e. mesh nodes to use.

    kwargs : dict
        Arguments for :meth:`matplotlib.axes.Axes.hist` if ``method`` is ``'histo'``,
        else for :meth:`matplotlib.axes.Axes.plot`.
    """
    ax.set_ylim(bottom=0)
    if not isinstance(samples,Mesh):
        mesh = samples.to_mesh([parameter],method=method,bins=bins)
    else:
        mesh = samples
    x,pdf = mesh([parameter])
    if normalize == 'max': pdf /= pdf.max()
    if method == 'histo':
        ax.hist(x,bins=mesh.edges[0],weights=pdf,density=False,**kwargs)
    else:
        ax.plot(x,pdf,**kwargs)


def plot_samples_2d(ax, samples, parameters, sigmas=2, method='gaussian_kde', scatter=False, bins=60, colors=None, fill=True, lighten=-0.5, **kwargs):
    """
    Plot 2D samples for parameter.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where to plot samples.

    samples : Samples
        Samples to plot.

    parameter : tuple, list, ParameterCollection
        The two parameters to plot samples for.

    sigmas : int, list, array, default=2
        Sigma levels to plot. If scalar, plot levels up to (including) ``sigmas``.

    method : string
        Method to interpolate samples on mesh, either:

        - 'gaussian_kde': Gaussian kernel density estimation, based on :class:`scipy.stats.gaussian_kde`
        - 'cic' : Cloud-in-Cell assignment
        - 'histo' : simple binning.

    scatter : bool, string, default=False
        Whether to plot samples as points.
        If ``'only'``, plot only points (and not contours).

    bins : int, default=60
        Number of bins i.e. mesh nodes to use.

    colors : list, string, tuple
        Single or list of colors.
        This list is extended to the number of required ``sigmas`` with ``ligthen``.

    fill : bool, default=True
        Whether to fill contours.

    lighten : float, default=-0.5
        Positive number ligthens colors, negative darkens colors.

    kwargs : dict
        Arguments for :meth:`matplotlib.axes.Axes.contourf` if ``fill`` is ``True``,
        else for :meth:`matplotlib.axes.Axes.coutour`.
    """
    parameters = list(parameters)
    if scatter:
        # scatter plot
        ax.scatter(samples[parameters[0]],samples[parameters[1]],**kwargs)
        if scatter == 'only': return # only scatter plot
    # contour plot
    if not isinstance(samples,Mesh):
        mesh = samples.to_mesh(parameters,method=method,bins=bins)
    else:
        mesh = samples
    levels = mesh.get_sigmas(sigmas)
    for ilevel,level in enumerate(levels):
        if level is None:
            levels = levels[:ilevel]
    (x,y),pdf = mesh(parameters)
    if colors is not None:
        colors = get_color_sequence(colors,ncolors=len(levels),lighten=lighten)
    if fill:
        ax.contourf(x,y,pdf.T,levels=levels[::-1]+[pdf.max()+1.],colors=colors,**kwargs)
    else:
        ax.contour(x,y,pdf.T,levels=levels[::-1],colors=colors,**kwargs)


def plot_normal_1d(ax, mean=0., covariance=1., lim=None, normalize='max', **kwargs):
    """
    Plot 1D normal distribution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where to plot distribution.

    mean : float, default=0
        Distribution mean.

    covariance : float, default=1
        Distribution covariance.

    lim : tuple, default=None
        x-limits. Defaults to current ``ax`` limits.

    normalize : string, default='max'
        How to normalize 1D distribution.
        Use 'max' to set maximum at 1 (most common),
        else distribution will be normalized as probability density (integral of 1).

    kwargs : dict
        Arguments for :meth:`matplotlib.axes.Axes.plot`.
    """
    if lim is None: lim = ax.get_xlim()
    x = np.linspace(*lim,num=1000)
    y = np.exp(-(x-mean)**2/2./covariance)
    if normalize != 'max': y *= 1./(covariance*np.sqrt(2.*np.pi))
    ax.plot(x,y,**kwargs)


def plot_contour_2d(ax, contours, sigmas=2, colors=None, fill=False, lighten=-0.5, **kwargs):
    """
    Plot 2D contours.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where to plot contours.

    contours : tuple, list, array
        List (or tuple) or arrays with x and y coordinates along first axis forming a contour in the x-y plane.
        Such array must be given for each confidence level.

    sigmas : int, list, array, default=2
        Sigma levels to plot. If scalar, plot levels up to (including) ``sigmas``.

    colors : list, string, tuple
        Single or list of colors.
        This list is extended to the number of required ``sigmas`` with ``ligthen``.

    fill : bool, default=True
        Whether to fill contours.

    lighten : float, default=-0.5
        Positive number ligthens colors, negative darkens colors.

    kwargs : dict
        Arguments for :meth:`matplotlib.axes.Axes.fill` if ``fill`` is ``True``,
        else for :meth:`matplotlib.axes.Axes.plot`.
    """
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


def plot_normal_2d(ax, mean=0, covariance=1., sigmas=2, **kwargs):
    r"""
    Plot 2D normal distribution.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes where to plot distribution.

    mean : float, array, default=0
        Distribution mean. If array, must be 1D.

    covariance : float, array, default=None
        Distribution covariance.
        If 1D array, must contain :math:`(\sigma_{xx}, \sigma_{yy}, \sigma_{xy})`
        If 2D array, covariance matrix.

    kwargs : dict
        Arguments for :func:`plot_contour_2d`.
    """
    if np.ndim(sigmas) == 0: sigmas = 1 + np.arange(sigmas)

    radii = np.sqrt(nsigmas_to_deltachi2(sigmas,ndof=2))
    t = np.linspace(0.,2.*np.pi,1000,endpoint=False)
    ct = np.cos(t)
    st = np.sin(t)
    mean_ = np.zeros(2,dtype='f8')
    mean_[:] = mean
    covariance = np.array(covariance)
    if covariance.size <= 1:
        sigx2 = sigy2 = covariance.flat[0]
        sigxy = 0.
    elif covariance.size == 3:
        sigx2,sigy2,sigxy = covariance
    else:
        sigx2,sigy2,sigxy = [covariance[i,j] for i,j in [(0,0),(1,1),(0,1)]]

    contours = []
    for radius in radii:
        a = radius * np.sqrt(0.5 * (sigx2 + sigy2) + np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
        b = radius * np.sqrt(0.5 * (sigx2 + sigy2) - np.sqrt(0.25 * (sigx2 - sigy2)**2. + sigxy**2.))
        th = 0.5 * np.arctan2(2. * sigxy, sigx2 - sigy2)
        x = mean_[0] + a * ct * np.cos(th) - b * st * np.sin(th)
        y = mean_[1] + a * ct * np.sin(th) + b * st * np.cos(th)
        contours.append((x,y))

    plot_contour_2d(ax, contours, **kwargs)


class ParameterPlotStyle(plotting.BasePlotStyle):

    """Base plotting class for parameter values."""

    def get_color(self, label, labels=None):
        """
        Return color corresponding to label ``label``.
        If :attr:`colors` is a list, return color at index of ``label`` in list of labels ``labels``.
        """
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

    @staticmethod
    def get_parameters(parameters, chains=None):
        """
        Return parameters.

        Parameters
        ----------
        parameters : tuple, list, ParameterCollection
            List of parameter (names).
            If parameter name, get the xorresponding parameter in the first element in ``chains`` that contains this parameter.

        chains : list
            List of classes with ``parameters`` attribute.

        Returns
        -------
        toret : Parameter, ParameterCollection
            Return single parameter if ``parameters`` is a single parameter (name).
        """
        if parameters is None:
            raise ValueError('Please provide parameter(s)')
        isscalar = not isinstance(parameters,(tuple,list,ParameterCollection))
        if isscalar:
            parameters = [parameters]
        toret = ParameterCollection(parameters)
        if chains is None:
            if isscalar:
                return toret[0]
            return toret
        elif not isinstance(chains,(tuple,list)):
            chains = [chains]
        for iparam,param in enumerate(toret):
            if isinstance(parameters[iparam],Parameter): # keep input parameter
                continue
            for ichain,chain in enumerate(chains): # if input was not parameter, take it from chains
                if param.name in chain.parameters:
                    toret[iparam] = chain.parameters[param.name]
                    break
        if isscalar:
            return toret[0]
        return toret

    def get_default_truths(self, parameters, truths=None):
        """
        Return default truth / reference values for parameters.

        Parameters
        ----------
        parameters : Parameter, tuple, list, ParameterCollection
            Parameters to return truth values for.

        truths : list, default=None
            List of truth values.
            If ``None``, return :attr:`truths` if exists, else set all truths to ``None``.
            If ``'value'``, return list of :attr:`Parameter.value`.

        Returns
        -------
        truths : float, list
            Return single value if ``parameters`` is a single parameter.
        """
        isscalar = not isinstance(parameters,(tuple,list,ParameterCollection))
        if isscalar:
            truths = [truths]
            parameters = [parameters]
        truths = self.get_list('truths',truths,[None]*len(parameters))
        toret = [param.value if truth == 'value' else truth for truth,param in zip(truths,parameters)]
        if isscalar:
            return toret[0]
        return toret


class SamplesPlotStyle(ParameterPlotStyle):
    """
    Plotting style for samples.
    It holds default attributes (:attr:`truths`, :attr:`colors`, etc.)
    that can be set at initialization (``style = SamplesPlotStyle(colors='r')``) or at any
    time using :meth:`update`.
    """
    def __init__(self, style=None, **kwargs):
        """
        Initialize :class:`SamplesPlotStyle`.

        Parameters
        ----------
        style : SamplesPlotStyle, default=None
            A plotting style to start from, which will be updated with ``kwargs``.

        kwargs : dict
            Attributes for :class:`SamplesPlotStyle`.
        """
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

    @staticmethod
    def get_default_parameters(parameters, samples):
        """If parameters is ``None``, return varied parameters from ``samples``."""
        if parameters is None:
            parameters = samples.columns(varied=True)
        return parameters

    def plot_1d(self, chains=None, parameter=None, labels=None, truth=None, ax=None, filename=None):
        """
        Plot 1D distribution of samples.

        Parameters
        ----------
        chains : list, default=None
            List of :class:`Samples` instances.
            Defaults to :attr:`chains`.

        parameter : Parameter, ParamName, string, tuple
            Parameter name.

        labels : list, string
            Label(s) for input chains.

        truth : float, string, default=None
            Plot this truth / reference value for parameter.
            If ``'value'``, take :attr:`Parameter.value`.

        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        tosave = ax is None
        if tosave: ax = plt.gca()
        chains = self.get_list('chains',chains)
        parameter = self.get_parameters(parameter,chains=chains)
        truth = self.get_default_truths(parameter,truth)
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
        """
        Plot 2D distribution of samples.

        Parameters
        ----------
        chains : list, default=None
            List of :class:`Samples` instances.
            Defaults to :attr:`chains`.

        parameters : tuple, list, ParameterCollection
            The two parameter name(s).

        labels : list, string
            Label(s) for input chains.

        truths : list, default=None
            Plot this truths / reference values for parameters.
            See :meth:`get_default_truths`.

        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        tosave = ax is None
        if tosave: ax = plt.gca()
        chains = self.get_list('chains',chains)
        parameters = self.get_parameters(parameters,chains=chains)
        truths = self.get_default_truths(parameters,truths)
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
        """
        Make corner plot.

        Parameters
        ----------
        chains : list, default=None
            List of :class:`Samples` instances.
            Defaults to :attr:`chains`.

        parameters : list, ParameterCollection
            The parameter names.

        labels : list, string
            Label(s) for input chains.

        truths : list, default=None
            Plot this truths / reference values for parameters.
            See :meth:`get_default_truths`.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure.

        dax : dict
            Dictionary of :class:`ParamName` and to :class:`matplotlib.axes.Axes`:
            name: axes for 1D plots and (name1, name2): axes for 2D plots.
        """
        chains = self.get_list('chains',chains)
        parameters = self.get_default_parameters(parameters,chains[0])
        parameters = self.get_parameters(parameters,chains=chains)
        truths = self.get_default_truths(parameters,truths)
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
        return fig, dax

    def plot_chain(self, chain, parameters=None, filename=None):
        """
        Make trace plot.

        Parameters
        ----------
        chain : Samples
            Samples to plot.

        parameters : list, ParameterCollection
            The parameter names.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure.

        lax : array
            Array of axes.
        """
        parameters = self.get_default_parameters(parameters,chain)
        parameters = self.get_parameters(parameters,chains=chain)
        nparams = len(parameters)
        steps = 1 + np.arange(chain.gsize)
        figsize = self.figsize or (8,1.5*nparams)
        fig,lax = plt.subplots(nparams,sharex=True,sharey=False,figsize=figsize,squeeze=False)
        lax = lax.flatten()

        for ax,param in zip(lax,parameters):
            ax.grid(True)
            ax.tick_params(labelsize=self.labelsize)
            ax.set_ylabel(param.get_label(),fontsize=self.labelsize)
            ax.plot(steps,chain.gget(param),color='k')

        lax[-1].set_xlabel('step',fontsize=self.labelsize)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return fig, lax

    def plot_gelman_rubin(self, chains=None, parameters=None, multivariate=False, threshold=1.1, slices=None, ax=None, filename=None):
        """
        Plot Gelman-Rubin statistics.

        Parameters
        ----------
        chains : list, default=None
            List of :class:`Samples` instances.
            Defaults to :attr:`chains`.

        parameters : list, ParameterCollection
            The parameter names.

        multivariate : bool, default=False
            If ``True``, add line for maximum of eigen value of Gelman-Rubin matrix.
            See :meth:`Samples.gelman_rubin`.

        threshold : float, default=1.1
            If not ``None``, plot horizontal line at this value.

        slices : list, array
            List of increasing number of steps to include in calculation of Gelman-Rubin statistics.

        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        chains = self.get_list('chains',chains)
        parameters = self.get_default_parameters(parameters,chains[0])
        parameters = self.get_parameters(parameters,chains=chains)
        if slices is None:
            nsteps = np.amin([chain.gsize for chain in chains])
            slices = np.arange(100,nsteps,500)
        gr_multi = []
        gr = {param.name:[] for param in parameters}
        for end in slices:
            chains_ = [chain.gslice(end) for chain in chains]
            if multivariate: gr_multi.append(Samples.gelman_rubin(chains_,parameters,method='eigen').max())
            for param in gr: gr[param].append(Samples.gelman_rubin(chains_,param,method='diag'))
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
        r"""
        Plot integrated autocorrelation time.

        Parameters
        ----------
        chains : list, default=None
            List of :class:`Samples` instances.
            Defaults to :attr:`chains`.

        parameters : list, ParameterCollection
            The parameter names.

        threshold : float, default=50
            If not ``None``, plot :math:`y = x/\mathrm{threshold}` line.
            Integrated autocorrelation time estimation can be considered reliable when falling under this line.

        slices : list, array
            List of increasing number of steps to include in calculation of autocorrelation time.

        ax : matplotlib.axes.Axes, default=None
            Axes where to plot samples. If ``None``, takes current axes.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        chains = self.get_list('chains',chains)
        parameters = self.get_default_parameters(parameters,chains[0])
        parameters = self.get_parameters(parameters,chains=chains)
        if slices is None:
            nsteps = np.amin([chain.gsize for chain in chains])
            slices = np.arange(100,nsteps,500)

        autocorr = {param.name:[] for param in parameters}
        weights = [chain.sum('metrics.weight') for chain in chains]
        for end in slices:
            chains_ = [chain.gslice(end) for chain in chains]
            for param in autocorr:
                tmp = Samples.integrated_autocorrelation_time(chains,param)
                autocorr[param].append(tmp)
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


class ProfilesPlotStyle(ParameterPlotStyle):
    """
    Plotting style for profiles.
    It holds default attributes (:attr:`truths`, :attr:`colors`, etc.)
    that can be set at initialization (``style = ProfilesPlotStyle(colors='r')``) or at any
    time using :meth:`update`.
    """
    def __init__(self, style=None, **kwargs):
        """
        Initialize :class:`ProfilesPlotStyle`.

        Parameters
        ----------
        style : ProfilesPlotStyle, default=None
            A plotting style to start from, which will be updated with ``kwargs``.

        kwargs : dict
            Attributes for :class:`ProfilesPlotStyle`.
        """
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

    @staticmethod
    def get_default_parameters(parameters, profiles):
        """If parameters is ``None``, return varied parameters from ``profiles``."""
        if parameters is None:
            parameters = profiles.parameters.select(varied=True)
        return parameters

    def plot_aligned(self, profiles=None, parameter=None, ids=None, labels=None, truth=None, yband=None, ax=None, filename=None):
        """
        Plot best fit estimates for single parameter.

        Parameters
        ----------
        profiles : list, default=None
            List of :class:`Profiles` instances.
            Defaults to :attr:`profiles`.

        parameter : Parameter, ParamName, string, tuple
            Parameter name.

        ids : list, string
            Label(s) for input profiles.

        labels : list, string
            Label(s) for best fits within each :class:`Profiles` instance.

        truth : float, string, default=None
            Plot this truth / reference value for parameter.
            If ``'value'``, take :attr:`Parameter.value`.

        yband : float, tuple, default=None
            If not ``None``, plot horizontal band.
            If tuple and last element set to ``'abs'``,
            absolute lower and upper y-coordinates of band;
            lower and upper fraction around truth.
            If float, fraction around truth.

        ax : matplotlib.axes.Axes, default=None
            Axes where to plot profiles. If ``None``, takes current axes.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        tosave = ax is None
        if tosave: ax = plt.gca()
        profiles = self.get_list('profiles',profiles)
        parameter = self.get_parameters(parameter,chains=profiles)
        truth = self.get_default_truths(parameter,truth)
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
            if np.ndim(yband) == 0:
                yband = (yband,yband)
            if yband[-1] == 'abs':
                low,up = yband[0],yband[1]
            else:
                if truth is None:
                    raise ValueError('Plotting relative band requires truth value.')
                low,up = truth*(1-yband[0]),truth*(1+yband[1])
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
        """
        Plot best fit estimates for several parameters.

        Parameters
        ----------
        profiles : list, default=None
            List of :class:`Profiles` instances.
            Defaults to :attr:`profiles`.

        parameters : list, ParameterCollection
            The parameter names.

        ids : list, string
            Label(s) for input profiles.

        labels : list, string
            Label(s) for best fits within each :class:`Profiles` instance.

        truths : list, default=None
            Plot this truths / reference values for parameters.
            See :meth:`get_default_truths`.

        ybands : list, default=None
            If not ``None``, plot horizontal bands.
            See :meth:`plot_aligned`.

        ylimits : list, default=None
            If not ``None``, limits  for y axis.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure.

        lax : array
            Array of axes.
        """
        profiles = self.get_list('profiles',profiles)
        parameters = self.get_default_parameters(parameters,profiles[0])
        truths = self.get_default_truths(parameters,truths)
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
        return fig,lax

    def _profiles_to_samples(self, profiles, parameters, select='min', residual='parabolic_errors', truths=None):
        # Turn :class:`Profiles` instances to :class:`Samples`
        # If residual is not ``None``, divide distance to truth (or mean if truth not provided) by corresponding error.
        profiles = self.get_list('profiles',profiles)
        if isinstance(profiles[0],Samples):
            return profiles

        def to_samples(profiles, parameters):
            parameters = self.get_default_parameters(parameters,profiles[0])
            toret = Profiles.to_samples(profiles,parameters=parameters,name='bestfit',select=select)
            if residual:
                for iparam,param in enumerate(parameters):
                    if truths is not None and truths[iparam] is not None:
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
        # Return label for parameter
        # If residual, make label as Delta param / sigma_param
        if residual:
            if parameter.latex is not None:
                return '$\Delta {0}/\sigma({0})$'.format(parameter.latex)
            return '$\Delta$ {0}/sigma({0})'.format(parameter.name)
        return parameter.get_label()

    def plot_1d(self, profiles=None, parameter=None, select='min', residual='parabolic_errors', truth=None, filename=None, **kwargs):
        """
        Plot 1D distribution of best fits.

        Parameters
        ----------
        profiles : list, default=None
            List of :class:`Profiles` instances.
            Defaults to :attr:`profiles`.

        parameter : Parameter, ParamName, string, tuple
            Parameter name.

        select : string, default='min'
            Rule to select the best fit in each :class:`Profiles` instance.

        residual : string, default='parabolic_errors'
            If not ``None``, divide distance to truth (or mean of truth is ``None``) by this quantity.

        truth : float, string, default=None
            Plot this truth / reference value for parameter.
            If ``'value'``, take :attr:`Parameter.value`.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        kwargs : dict
            Arguments for :meth:`SamplesPlotStyle.plot_1d`.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        truths = [self.get_default_truths(parameter,truth)]
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

    def plot_2d(self, profiles=None, parameters=None, select='min', residual='parabolic_errors', truths=None, filename=None, **kwargs):
        """
        Plot 2D distribution of best fits.

        Parameters
        ----------
        profiles : list, default=None
            List of :class:`Profiles` instances.
            Defaults to :attr:`profiles`.

        parameters : tuple, list, ParameterCollection
            The two parameter name(s).

        select : string, default='min'
            Rule to select the best fit in each :class:`Profiles` instance.

        residual : string, default='parabolic_errors'
            If not ``None``, divide distance to truth (or mean of truth is ``None``) by this quantity.

        truths : list, default=None
            Plot this truths / reference values for parameters.
            See :meth:`get_default_truths`.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        kwargs : dict
            Arguments for :meth:`SamplesPlotStyle.plot_2d`.

        Returns
        -------
        ax : matplotlib.axes.Axes
        """
        truths = self.get_default_truths(parameters,truths)
        chains = self._profiles_to_samples(profiles=profiles,parameters=parameters,select=select,residual=residual,truths=truths)
        parameters = self.get_parameters(parameters,chains=chains)
        ax = SamplesPlotStyle.plot_2d(self,chains,parameters=parameters,**kwargs)
        ax.set_xlabel(self._get_label(parameters[0],residual=residual))
        ax.set_ylabel(self._get_label(parameters[1],residual=residual))
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax

    def plot_corner(self, profiles=None, parameters=None, select='min', residual='parabolic_errors', truths=None, **kwargs):
        """
        Make corner plot of best fits.

        Parameters
        ----------
        profiles : list, default=None
            List of :class:`Profiles` instances.
            Defaults to :attr:`profiles`.

        parameters : tuple, list, ParameterCollection
            The two parameter name(s).

        select : string, default='min'
            Rule to select the best fit in each :class:`Profiles` instance.

        residual : string, default='parabolic_errors'
            If not ``None``, divide distance to truth (or mean of truth is ``None``) by this quantity.

        truths : list, default=None
            Plot this truths / reference values for parameters.
            See :meth:`get_default_truths`.

        filename : string, default=None
            If not ``None``, file name where to save figure.

        kwargs : dict
            Arguments for :meth:`SamplesPlotStyle.plot_corner`.

        Returns
        -------
        fig : matplotlib.figure.Figure
            Figure.

        dax : dict
            Dictionary of :class:`ParamName` and to :class:`matplotlib.axes.Axes`:
            name: axes for 1D plots and (name1, name2): axes for 2D plots.
        """
        truths = self.get_default_truths(parameters,truths)
        chains = self._profiles_to_samples(profiles=profiles,parameters=parameters,select=select,residual=residual,truths=truths)
        return SamplesPlotStyle.plot_corner(self,chains,parameters=parameters,**kwargs)
