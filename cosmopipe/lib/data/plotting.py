import re
import logging

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

from cosmopipe.lib import plotting, utils
from cosmopipe.lib.plotting import make_list
from cosmopipe.lib.utils import BaseClass
from .projection import ProjectionName


class BaseDataPlotStyle(plotting.BasePlotStyle):

    def __init__(self, style=None, **kwargs):
        super(BaseDataPlotStyle,self).__init__(style=style)
        self.projs = None
        self.xlabel = None
        self.ylabel = None
        self.color = None
        self.linestyles = '-'
        self.xscale = 'linear'
        self.yscale = 'linear'
        self.labelsize = 17
        self.ticksize = 15
        self.errorbar = 'fill'
        self.filename = None
        self.grid = True
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

    def get_color(self, proj, projs=None):
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
        return color

    @staticmethod
    def get_label(proj):
        return ProjectionName(proj).get_projlabel()

    def get_projs(self, data_vectors=None):
        data_vectors = self.get_list('data_vectors',value=data_vectors)
        return self.projs or data_vectors[0].get_projs()

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

    def plot(self, data_vectors=None, covariance=None, error_mean=None, ax=None, filename=None):
        if ax is None: ax = plt.gca()
        self.set_ax_attrs(ax)
        data_vectors = self.get_list('data_vectors',data_vectors)
        linestyles = self.get_list('linestyles',default=['-']*len(data_vectors))
        projs = self.get_projs(data_vectors)
        add_legend = data_vectors[0].has_proj()
        covariance = self.get('covariance',covariance)
        if covariance is not None:
            for iproj,proj in enumerate(projs):
                x = self.get_covx(covariance,proj=proj)
                if error_mean is not None:
                    xdata = self.get_x(data_vectors[error_mean],proj=proj)
                    ydata = self.get_y(data_vectors[error_mean],proj=proj)
                    y = np.interp(x,xdata,ydata)
                else:
                    y = self.get_covy(covariance,proj=proj)
                std = self.get_covstd(covariance,proj=proj)
                if self.errorbar == 'fill': ax.fill_between(x,y-std,y+std,alpha=0.5,facecolor=self.get_color(iproj,projs=projs),linewidth=0.0)
                else: ax.errorbar(x,y,std,fmt='none',color=self.get_color(iproj,projs=projs))
        for idata,data_vector in enumerate(data_vectors):
            for iproj,proj in enumerate(projs):
                label = self.get_label(proj) if idata == 0 else None
                ax.plot(self.get_x(data_vector,proj=proj),self.get_y(data_vector,proj=proj),label=label,linestyle=linestyles[idata],color=self.get_color(iproj,projs=projs),**self.kwplt)
        if add_legend:
            ax.legend(fontsize=self.labelsize)
        if self.grid:
            ax.grid(self.grid)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return ax


class PowerSpectrumPlotStyle(BaseDataPlotStyle):

    def __init__(self, style=None, **kwargs):
        super(PowerSpectrumPlotStyle,self).__init__(style=style)
        self.xlabel = '$k$ [$h \ \\mathrm{Mpc}^{-1}$]'
        self.ylabel = '$k P(k)$ [$(\\mathrm{Mpc} \ h)^{-1})^{2}$]'
        self.update(**kwargs)

    def get_projs(self, data_vectors=None):
        projs = super(PowerSpectrumPlotStyle,self).get_projs(data_vectors=data_vectors)
        return [proj for proj in projs if proj.space in (None,ProjectionName.POWER)]

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

    def get_projs(self, data_vectors=None):
        projs = super(CorrelationFunctionPlotStyle,self).get_projs(data_vectors=data_vectors)
        return [proj for proj in projs if proj.space in (None,ProjectionName.CORRELATION)]

    @staticmethod
    def get_y(data, *args, **kwargs):
        return data.get_x(*args,**kwargs)**2*data.get_y(*args,**kwargs)

    @staticmethod
    def get_covy(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]**2*covariance.get_y(*args,**kwargs)[0]

    @staticmethod
    def get_covstd(covariance, *args, **kwargs):
        return covariance.get_x(*args,**kwargs)[0]**2*covariance.get_std(*args,**kwargs)


def DataPlotStyle(style=None, **kwargs):

    if isinstance(style, plotting.BasePlotStyle):
        style.update(**kwargs)
        return style

    if style is None:
        data_vectors = kwargs.get('data_vectors',None)
        if data_vectors is not None:
            projs = make_list(data_vectors)[0].get_projs()
            spaces = np.unique([proj.space for proj in projs])
            if len(spaces) == 1:
                style = spaces[0]
            else:
                raise ValueError('Data vector contains {}. Specify the desired type'.format(spaces))

    return dataplotstyle_registry[style](**kwargs)


dataplotstyle_registry = {None:plotting.BasePlotStyle}
dataplotstyle_registry[ProjectionName.POWER] = PowerSpectrumPlotStyle
dataplotstyle_registry[ProjectionName.CORRELATION] = CorrelationFunctionPlotStyle


class CovarianceMatrixPlotStyle(plotting.BasePlotStyle):

    def __init__(self, style=None, data_styles=None, **kwargs):
        super(CovarianceMatrixPlotStyle,self).__init__(style=style)
        self.data_styles = make_list(data_styles,2)
        self.wspace = self.hspace = 0.18
        self.figsize = None
        self.ticksize = 13
        self.norm = None
        self.barlabel = None
        self.update(**kwargs)

    def get_styles(self, covariance=None):
        if covariance is not None:
            return tuple(DataPlotStyle(style=style,data_vectors=data) for style,data in zip(self.data_styles,covariance.x))
        return tuple(DataPlotStyle(style) for style in data_styles)

    @staticmethod
    def get_mat(covariance):
        return covariance.get_cov()

    def plot(self, covariance=None, filename=None):
        covariance = self.get('covariance',covariance)
        mat = self.get_mat(covariance)
        norm = self.norm or Normalize(vmin=mat.min(),vmax=mat.max())
        styles = self.get_styles(covariance)
        for s in styles: s.projs = s.get_projs()
        x = [[x.get_x(proj=proj) for proj in s.projs] for s,x in zip(styles,covariance.x)]
        mat = [[mat[np.ix_(*covariance.get_index(proj=(proj1,proj2)))] for proj1 in styles[0].projs] for proj2 in styles[1].projs]
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
                iproj,jproj = styles[0].projs[i],styles[1].projs[j]
                mesh = ax.pcolor(x[0][i],x[1][j],mat[i][j].T,norm=norm,cmap=plt.get_cmap('jet_r'))
                if i>0: ax.yaxis.set_visible(False)
                elif jproj: ax.set_ylabel(jproj.get_xlabel())
                if j>0: ax.xaxis.set_visible(False)
                elif iproj: ax.set_xlabel(iproj.get_xlabel())
                ax.tick_params(labelsize=self.ticksize)
                label1,label2 = styles[0].get_label(styles[0].projs[i]),styles[1].get_label(styles[1].projs[j])
                if label1 is not None or label2 is not None:
                    text = '{}\nx {}'.format(label1,label2)
                    ax.text(0.05,0.95,text,horizontalalignment='left',verticalalignment='top',\
                            transform=ax.transAxes,color='black',fontsize=styles[0].labelsize)

        #plotting.suplabel('x',styles[0].xlabel,shift=0,labelpad=17,size=styles[0].labelsize)
        #plotting.suplabel('y',styles[1].xlabel,shift=0,labelpad=17,size=styles[0].labelsize)
        fig.subplots_adjust(right=xextend)
        cbar_ax = fig.add_axes([xextend+0.05,0.15,0.03,0.7])
        cbar_ax.tick_params(labelsize=self.ticksize)
        cbar = fig.colorbar(mesh,cax=cbar_ax)
        cbar.set_label(self.barlabel,fontsize=styles[0].labelsize,rotation=90)
        filename = filename or self.filename
        if filename: self.savefig(filename)
        return lax,cbar_ax


class CorrelationMatrixPlotStyle(CovarianceMatrixPlotStyle):

    @staticmethod
    def get_mat(covariance):
        return covariance.get_corrcoef()


def MatrixPlotStyle(style=None, **kwargs):

    if isinstance(style, plotting.BasePlotStyle):
        style.update(**kwargs)
        return style

    return matrixplotstyle_registry[style](**kwargs)


matrixplotstyle_registry = {}
matrixplotstyle_registry['cov'] = CovarianceMatrixPlotStyle
matrixplotstyle_registry['corr'] = CorrelationMatrixPlotStyle
