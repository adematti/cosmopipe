import logging
import itertools

import numpy as np
from scipy import special

from cosmopipe.lib.data_vector import CovarianceMatrix
from .base import ProjectionName, ProjectionBase, ProjectionBaseCollection, ModelCollection
from .evaluation import ModelEvaluation
from .hankel_transform import HankelTransform


def legendre_product_integral(ells, range=None, norm=False):
    poly = 1
    for ell in ells:
        poly *= special.legendre(ell)
    integ = poly.integ()
    if range is None:
        range = (-1,1)
    toret = integ(range[-1]) - integ(range[0])
    if norm:
        toret /= (range[-1] - range[0])
    return toret


def weights_trapz(x):
    return np.concatenate([[x[1]-x[0]],x[2:]-x[:-2],[x[-1]-x[-2]]])/2.


def symproduct_iterator(x,y):

    for (ix1,x1),(ix2,x2) in itertools.product(x,y):
        if x1 > x2: continue
        coeff = 1 + ix2 != ix1
        yield coeff,(ix1,x1),(ix2,x2)


def interval_intersection(*intervals):
    return (max(interval[0] for interval in intervals),min(interval[1] for interval in intervals))


def empty_interval(interval):
    return interval[0] >= interval[1]


class GaussianCovarianceMatrix(CovarianceMatrix):

    logger = logging.getLogger('GaussianCovarianceMatrix')

    def __init__(self, data, model_base=None, volume=None, xnum=3, munum=100, integration=None, kcutoff=(1e-6,1e1), attrs=None):

        self.projs = data.get_projs()
        self.model_bases = ProjectionBaseCollection(model_base)
        self.power_bases = self.model_bases.select(*[{'name':proj.name,'space':ProjectionBase.POWER} for proj in self.projs])

        # if no model provided for correlation, compute from power
        correlation_bases = {}
        for proj in self.projs:
            try:
                base = self.model_bases.get_by_proj(proj)
            except IndexError:
                if proj.space == ProjectionBase.CORRELATION:
                    proj = proj.copy(space=ProjectionBase.POWER)
                    new = self.model_bases.get_by_proj(proj)
                    if new not in correlation_bases:
                        correlation_bases[new] = []
                    correlation_bases[new].append(proj)

        self.ells = (0,2,4)
        ells = [list(base.projs) for base in self.power_bases if base.mode == ProjectionBase.MULTIPOLE]
        if ells:
            self.ells = ells[0]
            # intersection of ells
            for ells_ in ells[1:]:
                for ill,ell in enumerate(self.ells):
                    if ell not in ells_:
                        del self.ells[ill]

        self.ext_models = ModelCollection()
        for input_base,projs in correlation_bases.items():
            ells = self.ells
            if input_base.mode == ProjectionBase.MUWEDGE:
                poles = [proj.proj for proj in projs if proj.mode == ProjectionName.MULTIPOLE]
                if poles:
                    ells = range(0,max(poles)+1,2)
            model = HankelTransform(model=None,base=input_base,ells=ells)
            self.ext_models.set(model)
            self.model_bases.set(model.base)

        self.edges = []
        for proj in self.projs:
            edges = data.get_edges(proj=proj)[0]
            #print(proj,edges)
            self.edges.append(np.vstack([edges[:-1],edges[1:]]).T)

        self.evaluation = ModelEvaluation(data,model_base=self.model_bases,integration=integration)

        self.xnum = xnum
        self.munum = munum
        self.k = np.mean([base.x for base in self.power_bases],axis=0)
        self.k = self.k[(self.k > kcutoff[0]) & (self.k < kcutoff[-1])]

        self.attrs = attrs or {}
        if isinstance(volume,dict):
            volume = {frozenset(key):val for key,val in volume.items()}
        self.attrs.setdefault('volume',volume)

    def compute(self, models):

        models = ModelCollection(models)
        for extmodel in self.ext_models.models:
            extmodel.input_model = models.get(extmodel.input_base)

        models = models + self.ext_models
        mean = self.evaluation.to_data_vector(models)

        cov = [[None for proj in self.projs] for proj in self.projs]
        self._sigmak = {}

        for (iproj1,proj1),(iproj2,proj2) in itertools.product(enumerate(self.projs),enumerate(self.projs)):
            if iproj1 > iproj2:
                cov[iproj1][iproj2] = cov[iproj2][iproj1].T
                continue
            auto = proj2.fields == proj1.fields
            if auto:
                lfields = [proj1.fields]
            else:
                lfields = [(proj1.fields[0],proj2.fields[0]),(proj1.fields[1],proj2.fields[1]),(proj1.fields[0],proj2.fields[1]),(proj1.fields[1],proj2.fields[0])]
            pks = []
            self.model_base = ProjectionBase(projs=self.ells)

            def dummy(*args,**kwargs):
                return 0.

            for fields in lfields:

                try:
                    base = models.bases().get_by_proj(fields=fields,space=ProjectionBase.POWER)
                except IndexError:
                    if fields in [proj1.fields,proj2.fields]:
                        raise
                    pk = dummy
                else:
                    pk = models.get(base)

                if self.model_base.mode is None:
                    self.model_base.mode = base.mode
                elif self.model_base.mode != base.mode:
                    raise ValueError('Input power spectrum models should all be either multipoles or function of (k,mu)')
                pks.append(pk)


            pk2 = self.get_pk2(pks=pks)
            volume = self.attrs['volume']
            if isinstance(volume,dict):
                vol1 = volume[frozenset(proj1.fields)]
                vol2 = volume[frozenset(proj2.fields)]
                vol12 = volume[frozenset(proj1.fields + proj2.fields)]
                volume = vol12/(vol1*vol2)

            cov[iproj1][iproj2] = self.eval(pk2,proj1=proj1,proj2=proj2,edges1=self.edges[iproj1],edges2=self.edges[iproj2],volume=volume)

        self._sigmak = {}
        covariance = np.bmat(cov).A
        covariance = (covariance + covariance.T)/2.
        super(GaussianCovarianceMatrix,self).__init__(covariance,first=mean,attrs=self.attrs)


    def get_pk2(self, pks):

        # arXiv: 2007.09011 eq. 2 and 3.
        auto = len(pks) == 1

        if self.model_base.mode == ProjectionBase.MUWEDGE:

            if auto:

                def pk2(k, mu, **kwargs):
                    return pks[0](k,mu,**kwargs)**2

            else:

                def pk2(k, mu, **kwargs):
                    return 1./2.*(pks[0](k,mu,**kwargs)*pks[1](k,mu,**kwargs) + pks[2](k,mu,**kwargs)*pks[3](k,mu,**kwargs))


        if self.model_base.mode == ProjectionBase.MULTIPOLE:

            if auto:

                def pk2(k, **kwargs):
                    pkell = pks[0](k,**kwargs)
                    return {(ell1,ell2): coeff*pkell[:,ill1]*pkell[:,ill2] for coeff,(ill1,ell1),(ill2,ell2)
                                in symproduct_iterator(enumerate(self.model_base.projs),enumerate(self.model_base.projs))}

            else:

                def pk2(k, **kwargs):
                    pkells = [pk(k,**kwargs) for pk in pks]
                    return {(ell1,ell2): 1./2.*(pkells[0][:,ill1]*pkells[1][:,ill2] + pkells[2][:,ill1]*pkells[3][:,ill2]) for (ill1,ell1),(ill2,ell2)
                                in itertools.product(enumerate(self.model_base.projs),enumerate(self.model_base.projs))}

        return pk2

    def sigma_k(self, pk2, k=None, proj1=None, proj2=None, volume=None):

        store = k is None
        if store:
            if (proj1.proj,proj2.proj) in self._sigmak:
                return self._sigmak[proj1.proj,proj2.proj]
            k = self.k

        ells, muwedges = [0,0], [(0.,1.),(0.,1.)]
        for iproj,proj in enumerate([proj1,proj2]):
            if proj.mode == ProjectionName.MULTIPOLE: ells[iproj] = proj.proj
            elif proj.mode == ProjectionName.MUWEDGE: muwedges[iproj] = proj.proj
        dmu2 = np.prod([muwedge[1] - muwedge[0] for muwedge in muwedges])
        muwedge = interval_intersection(*muwedges)
        ell1,ell2 = ells

        if empty_interval(muwedge):
            return np.zeros_like(k)

        if self.model_base.mode == ProjectionBase.MUWEDGE:

            mu = np.linspace(muwedge[0],muwedge[1],self.munum)
            integrand = 2.*(2.*ell1+1.)*(2.*ell2+1.)/dmu2 * pk2(k,mu,grid=True) * special.legendre(ell1)(mu) * special.legendre(ell2)(mu)
            toret = np.trapz(integrand,x=mu,axis=-1)

        if self.model_base.mode == ProjectionBase.MULTIPOLE:

            pk2ell = pk2(k)
            toret = 0.
            for (ell1,ell2) in pk2ell:
                toret += pk2ell[ell1,ell2] * legendre_product_integral([ell1,ell2] + ells,range=muwedge)
            toret = 2.*(2.*ell1+1.)*(2.*ell2+1.)/dmu2 * toret

        toret *= 1./volume*k**2
        if store and (proj1.name,proj2.name) == (ProjectionBase.MULTIPOLE,ProjectionBase.MULTIPOLE):
            self._sigmak[proj2,proj1] = self._sigmak[proj1.proj,proj2.proj] = toret

        return toret

    def bin_integ(self, proj1=None, proj2=None, edges1=None, edges2=None):
        edges = edges1,edges2

        if (proj1.space,proj2.space) == (ProjectionName.POWER,ProjectionName.POWER):
            edge = interval_intersection(*edges)
            if empty_interval(edge):
                return None,None
            x = np.linspace(edge[0],edge[1],self.xnum)
            #x = x[x>self.k[0]]
            w = weights_trapz(x)
            dv = np.sum(w*x**2,axis=0)
            return x,w/dv**2

        x = tuple([np.linspace(edge[0],edge[1],self.xnum) for edge in edges])
        #x = tuple([x[x>self.k[0]] if proj.space == ProjectionName.POWER else x for x,proj in zip(x,[proj1,proj2])])
        w = tuple(weights_trapz(x_) for x_ in x)
        dv = tuple(np.sum(w_*x_**2,axis=0) for w_,x_ in zip(w,x))

        if (proj1.space,proj2.space) == (ProjectionName.POWER,ProjectionName.CORRELATION):
            (k,s),(wk,ws),(dvk,dvs) = x,w,dv
            ws = np.sum(s[:,None]**2*special.spherical_jn(proj2.proj,s[:,None]*k[None,:])*ws[:,None],axis=0)
            return k,wk/dvk,ws/dvs

        if (proj1.space,proj2.space) == (ProjectionName.CORRELATION,ProjectionName.CORRELATION):
            (s1,s2),(w1,w2),(dv1,dv2) = x,w,dv
            w1,w2 = (np.sum(s[:,None]**2*special.spherical_jn(proj.proj,s[:,None]*self.k[None,:])*w[:,None],axis=0) for s,w,proj in zip(x,w,(proj1,proj2)))
            return w1/dv1,w2/dv2


    def eval(self, pk2, proj1=None, proj2=None, edges1=None, edges2=None, volume=None):

        nindices = [len(edges) for edges in [edges1,edges2]]
        toret = np.zeros(nindices,dtype='f8')

        allowed = [ProjectionName.POWER,ProjectionName.CORRELATION]
        for proj in [proj1,proj2]:
            if proj.space not in allowed:
                raise ValueError('Required projection space is {}, but allowed spaces are {}'.format(proj.space,allowed))

        if (proj1.space,proj2.space) == (ProjectionName.CORRELATION,ProjectionName.POWER):
            return self.eval(pk2,proj1=proj2,proj2=proj1,edges1=edges2,edges2=edges1,volume=volume).T

        if (proj1.space,proj2.space) == (ProjectionName.POWER,ProjectionName.POWER):

            def sigma_integk(i1, i2):
                k,w = self.bin_integ(proj1,proj2,edges1[i1],edges2[i2])
                if k is None:
                    return 0.
                coeff = 2.*np.pi**2
                sigmak = self.sigma_k(pk2,k,proj1,proj2,volume=volume)
                return coeff*np.sum(w * sigmak)

            auto = proj2 == proj1
            for i1,i2 in itertools.product(range(nindices[0]),range(nindices[1])):
                if auto and i1 > i2:
                    toret[i1,i2] = toret[i2,i1]
                    continue
                toret[i1,i2] = sigma_integk(i1,i2)

            return toret

        elif (proj1.space,proj2.space) == (ProjectionName.POWER,ProjectionName.CORRELATION) and (proj1.mode,proj2.mode) == (ProjectionName.MULTIPOLE,)*2:

            def sigma_integk(i1, i2):
                coeff = np.array(1j**proj2.proj)
                if not np.iscomplex(coeff):
                    coeff = coeff.real
                k,wk,ws = self.bin_integ(proj1,proj2,edges1[i1],edges2[i2])
                sigmak = self.sigma_k(pk2,k,proj1,proj2,volume=volume)
                return coeff*np.sum(wk * ws * sigmak)

            for i1,i2 in itertools.product(range(nindices[0]),range(nindices[1])):
                #if i1 > i2:
                #    toret[i1,i2] = toret[i2,i1]
                #    continue
                toret[i1,i2] = sigma_integk(i1,i2)


        elif (proj1.space,proj2.space) == (ProjectionName.CORRELATION,ProjectionName.CORRELATION) and (proj1.mode,proj2.mode) == (ProjectionName.MULTIPOLE,)*2:

            def sigma_integk(i1, i2):
                coeff = np.array(1j**(proj1.proj + proj2.proj))/2./np.pi**2
                if not np.iscomplex(coeff):
                    coeff = coeff.real
                w1,w2 = self.bin_integ(proj1,proj2,edges1[i1],edges2[i2])
                sigmak = self.sigma_k(pk2,None,proj1,proj2,volume=volume)
                return coeff*np.sum(w1 * w2 * sigmak * weights_trapz(self.k))

            auto = proj2 == proj1
            for i1,i2 in itertools.product(range(nindices[0]),range(nindices[1])):
                if auto and i1 > i2:
                    toret[i1,i2] = toret[i2,i1]
                    continue
                toret[i1,i2] = sigma_integk(i1,i2)

        else: # power x correlation, correlation x correlation, muwedge x poles or muwedge x muwedge
            ells = range(0,2*max(self.ells),2)
            for ell1,ell2 in itertools.product(ells,ells):
                coeff = 1.
                if proj1.mode == ProjectionName.MULTIPOLE and ell1 != proj1.proj: continue
                if proj2.mode == ProjectionName.MULTIPOLE and ell2 != proj2.proj: continue
                if proj1.mode == ProjectionName.MUWEDGE: coeff *= legendre_product_integral([ell1],range=proj1.proj,norm=True)
                if proj2.mode == ProjectionName.MUWEDGE: coeff *= legendre_product_integral([ell2],range=proj2.proj,norm=True)
                #print(proj1.mode, proj2.mode, ell1, ell2, coeff)
                nproj1, nproj2 = proj1.copy(), proj2.copy()
                nproj1.mode, nproj2.mode = ProjectionName.MULTIPOLE, ProjectionName.MULTIPOLE
                nproj1.proj, nproj2.proj = ell1, ell2
                toret += coeff * self.eval(pk2,nproj1,nproj2,edges1=edges1,edges2=edges2,volume=volume)

        return toret
