import re
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from cosmopipe.lib import plotting, utils
from cosmopipe.lib.utils import BaseClass


class BaseDataPlotStyle(plotting.BasePlotStyle):

    def __init__(self, style=None, **kwargs):
        super(BaseDataPlotStyle,self).__init__(style=style)
        self.projs = None
        self.xlabel = None
        self.ylabel = None
        self.color = None
        self.linestyle = '-'
        self.xscale = 'linear'
        self.yscale = 'linear'
        self.labelsize = 17
        self.ticksize = 15
        self.filename = None
        self.kwplt = {}
        self.update(**kwargs)

    def update(self, **kwargs):
        for key,val in kwargs.items():
            setattr(self,key,val)

    def set_ax_attrs(self, ax):
        ax.set_xlabel(self.xlabel,fontsize=self.labelsize)
        ax.set_ylabel(self.ylabel,fontsize=self.labelsize)
        ax.set_xscale(self.xscale)
        ax.set_yscale(self.yscale)
        ax.tick_params(labelsize=self.ticksize)

    def get_kwplt(self, proj, projs=None):
        if projs is None:
            projs = self.projs
            index = self.projs.index(proj)
        else:
            index = proj
            proj = projs[index]
        if self.color is None:
            color = 'C{:d}'.format(index)
        elif isinstance(self.color,str):
            color = self.color
        elif isinstance(self.color,dict):
            color = self.color[proj]
        else:
            color = self.color[index]
        if isinstance(self.linestyle,str):
            linestyle = self.linestyle
        elif isinstance(self.linestyle,dict):
            linestyle = self.linestyle[proj]
        else:
            linestyle = self.linestyle[index]
        return {**{'color':color,'linestyle':linestyle},**self.kwplt}

    @staticmethod
    def get_label(proj):
        if proj is None:
            return None
        proj = str(proj)
        match = re.match('ell_(.*)',proj)
        if match:
            return '$\\ell = {}$'.format(match.group(1))
        return '${}$'.format(utils.txt_to_latex(proj))

    @staticmethod
    def get_x(data, *args, **kwargs):
        return data.get_x(*args,**kwargs)

    @staticmethod
    def get_y(data, *args, **kwargs):
        return data.get_y(*args,**kwargs)

    @staticmethod
    def get_covx(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]

    @staticmethod
    def get_covy(covariance, *args, **kwargs):
        return covariance.get_y(*args,**kwargs)[0]

    @staticmethod
    def get_covstd(covariance, *args, **kwargs):
        return covariance.get_std(*args,**kwargs)

    def plot(self, data_vectors, covariance=None, ax=None, filename=None):
        if ax is None: ax = plt.gca()
        self.set_ax_attrs(ax)
        if not isinstance(data_vectors,list):
            data_vectors = [data_vectors]
        add_legend = self.projs or data_vectors[0].projs
        projs = add_legend or [None]
        if covariance is not None:
            for iproj,proj in enumerate(projs):
                x = self.get_covx(covariance,proj=proj)
                y = self.get_covy(covariance,proj=proj)
                std = self.get_covstd(covariance,proj=proj)
                ax.fill_between(x,y-std,y+std,alpha=0.5,color=self.get_kwplt(iproj,projs=projs)['color'])
        for idata,data_vector in enumerate(data_vectors):
            for iproj,proj in enumerate(projs):
                label = self.get_label(proj) if idata == 0 else None
                ax.plot(self.get_x(data_vector,proj=proj),self.get_y(data_vector,proj=proj),label=label,**self.get_kwplt(iproj,projs=projs))
        if add_legend:
            ax.legend(fontsize=self.labelsize)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax


class PowerSpectrumPlotStyle(BaseDataPlotStyle):

    def __init__(self, style=None, **kwargs):
        super(PowerSpectrumPlotStyle,self).__init__(style=style)
        self.xlabel = '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
        self.ylabel = '$k P(k)$ [$(\\mathrm{Mpc} \ h)^{-1})^{2}$]'
        self.update(**kwargs)

    @staticmethod
    def get_y(data, *args, **kwargs):
        return data.get_x(*args,**kwargs)*data.get_y(*args,**kwargs)

    @staticmethod
    def get_covy(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]*covariance.get_y(*args,**kwargs)[0]

    @staticmethod
    def get_covstd(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]*covariance.get_std(*args,**kwargs)


class CorrelationFunctionPlotStyle(BaseDataPlotStyle):

    def __init__(self, style=None, **kwargs):

        super(CorrelationFunctionPlotStyle,self).__init__(style=style)
        self.xlabel = '$s$ [$\\mathrm{Mpc} \ h$]'
        self.ylabel = '$s^{2} \\xi(s)$ [$(\\mathrm{Mpc} \ h)^{-1})^{2}$]'
        self.update(**kwargs)

    @staticmethod
    def get_y(data, *args, **kwargs):
        return data.get_x(*args,**kwargs)**2*data.get_y(*args,**kwargs)

    @staticmethod
    def get_covy(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]**2*covariance.get_y(*args,**kwargs)[0]

    @staticmethod
    def get_covstd(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]**2*covariance.get_std(*args,**kwargs)


class CovarianceMatrixPlotStyle(plotting.BasePlotStyle):

    def __init__(self, style=None, data_styles=None, **kwargs):
        super(CovarianceMatrixPlotStyle,self).__init__(style=style)
        if not isinstance(data_styles,(tuple,list)):
            data_styles = (data_styles,data_styles)
        self.styles = tuple(PlotStyle(style) for style in data_styles)
        self.wspace = self.hspace = 0.18
        self.figsize = None
        self.ticksize = 13
        self.norm = None
        self.barlabel = None
        self.update(**kwargs)

    @staticmethod
    def get_mat(covariance):
        return covariance.get_cov()

    def plot(self, covariance, filename=None):
        mat = self.get_mat(covariance)
        norm = self.norm or Normalize(vmin=mat.min(),vmax=mat.max())
        for s,x in zip(self.styles,covariance.x):
            s.projs = s.projs or x.projs or [None]
        x = [[x.get_x(proj=proj) for proj in s.projs] for s,x in zip(self.styles,covariance.x)]
        mat = [[mat[np.ix_(*covariance.get_index(proj=(proj1,proj2)))] for proj1 in self.styles[0].projs] for proj2 in self.styles[1].projs]
        nrows = len(x[1])
        ncols = len(x[0])
        width_ratios = list(map(len,x[1]))
        height_ratios = list(map(len,x[0]))
        figsize = self.figsize or np.clip(sum(width_ratios)/7.,6,10)
        xextend = 0.8
        fig,lax = plt.subplots(nrows=nrows,ncols=ncols,sharex=False,sharey=False,
                                figsize=(figsize/xextend,figsize),
                                gridspec_kw={'width_ratios':width_ratios,'height_ratios':height_ratios},squeeze=False)
        fig.subplots_adjust(wspace=self.wspace,hspace=self.hspace)
        for i in range(ncols):
            for j in range(nrows):
                ax = lax[nrows-1-j][i]
                mesh = ax.pcolor(x[0][i],x[1][j],mat[i][j].T,norm=norm,cmap=plt.get_cmap('jet_r'))
                if i>0: ax.yaxis.set_visible(False)
                if j>0: ax.xaxis.set_visible(False)
                ax.tick_params(labelsize=self.ticksize)
                label1,label2 = self.styles[0].get_label(self.styles[0].projs[i]),self.styles[1].get_label(self.styles[0].projs[j])
                if label1 is not None or label2 is not None:
                    text = '{}\nx {}'.format(label1,label2)
                    ax.text(0.05,0.95,text,horizontalalignment='left',verticalalignment='top',\
                            transform=ax.transAxes,color='black',fontsize=self.styles[0].labelsize)

        plotting.suplabel('x',self.styles[0].xlabel,shift=0,labelpad=17,size=self.styles[0].labelsize)
        plotting.suplabel('y',self.styles[1].xlabel,shift=0,labelpad=17,size=self.styles[0].labelsize)
        fig.subplots_adjust(right=xextend)
        cbar_ax = fig.add_axes([xextend+0.05,0.15,0.03,0.7])
        cbar_ax.tick_params(labelsize=self.ticksize)
        cbar = fig.colorbar(mesh,cax=cbar_ax)
        cbar.set_label(self.barlabel,fontsize=self.styles[0].labelsize,rotation=90)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return lax,cbar_ax


class CorrelationMatrixPlotStyle(CovarianceMatrixPlotStyle):

    @staticmethod
    def get_mat(covariance):
        return covariance.get_corrcoef()


def DataPlotStyle(style, **kwargs):

    if isinstance(style, plotting.BasePlotStyle):
        style.update(**kwargs)
        return style

    return plotstyle_registry[style](**kwargs)


plotstyle_registry = {None:plotting.BasePlotStyle}
plotstyle_registry['pk'] = PowerSpectrumPlotStyle
plotstyle_registry['xi'] = CorrelationFunctionPlotStyle
plotstyle_registry['cov'] = CovarianceMatrixPlotStyle
plotstyle_registry['corr'] = CorrelationMatrixPlotStyle
