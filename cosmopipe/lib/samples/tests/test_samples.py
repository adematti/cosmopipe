import os

import numpy as np
from cosmopipe.lib.mpi import CurrentMPIComm

from cosmopipe.lib import setup_logging
from cosmopipe.lib.parameter import Parameter, ParamName, Prior
from cosmopipe.lib.samples import Samples, SamplesPlotStyle
from cosmopipe.lib.samples.plotting import plot_normal_1d, plot_normal_2d


def get_chains(parameters, n=4, size=4000):
    rng = np.random.RandomState(seed=42)
    ndim = len(parameters)
    mean = np.zeros(ndim,dtype='f8')
    cov = np.eye(ndim,dtype='f8')
    cov += 0.1 # off-diagonal
    invcov = np.linalg.inv(cov)
    chains = []
    for ichain in range(n):
        array = rng.multivariate_normal(mean,cov,size=size)
        diff = array-mean
        samples = Samples.from_array(array.T,columns=parameters,mpistate='broadcast')
        samples['metrics','logposterior'] = -1./2.*np.sum(diff.dot(invcov)*diff,axis=-1)
        for par in parameters:
            samples.parameters[par].fixed = False
        chains.append(samples)
    return mean,cov,chains


def iterate_mpi(chains):
    for chain in chains:
        chain.mpi_scatter()
    print('scattered')
    yield chains
    for chain in chains:
        chain.mpi_gather()
    print('gathered')
    yield chains
    toret = []
    print('broadcast')
    for chain in chains:
        toret.append(Samples.mpi_broadcast(chain))
    yield toret


def test_plotting():

    plot_dir = '_plots'
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    mean,cov,chains = get_chains(parameters,n=4,size=4000)

    for chains in iterate_mpi(chains):
        for method in ['histo','cic','gaussian_kde']:
            style = SamplesPlotStyle(kwplt_2d={'method':method})
            style.plot_1d(chains=chains[0],parameter='parameters.a',filename=os.path.join(plot_dir,'pdf_{}_a.png'.format(method)))
            style.plot_2d(chains=chains[0],parameters=['parameters.a','parameters.b'],truths=mean[:2],filename=os.path.join(plot_dir,'pdf_{}_ab.png'.format(method)))

            style.plot_corner(chains=chains[:2],parameters=parameters,truths=mean,labels=['1','2'],filename=os.path.join(plot_dir,'corner_{}.png'.format(method)))
            ax = style.plot_1d(chains=chains[0],parameter='parameters.a')
            plot_normal_1d(ax,mean=mean[0],covariance=cov[0,0],color='g')
            style.savefig(filename=os.path.join(plot_dir,'pdfg_{}_a.png'.format(method)))
            ax = style.plot_2d(chains=chains[0],parameters=['parameters.a','parameters.b'],truths=mean[:2])
            plot_normal_2d(ax,mean=mean[:2],covariance=cov[:2,:2],fill=False,colors='g')
            style.savefig(filename=os.path.join(plot_dir,'pdfg_{}_ab.png'.format(method)))
            style.fills = [True,False]
            fig,dax = style.plot_corner(chains=chains[:2],parameters=parameters,truths=mean,labels=['1','2'])
            plot_normal_1d(dax['parameters.a'],mean=mean[0],covariance=cov[0,0],color='g')
            plot_normal_2d(dax['parameters.a','parameters.b'],mean=mean[:2],covariance=cov[:2,:2],colors='g')
            style.savefig(filename=os.path.join(plot_dir,'cornerg_{}.png'.format(method)),fig=fig)

        style = SamplesPlotStyle()
        style.plot_chain(chains[0],parameters=['parameters.a'],filename=os.path.join(plot_dir,'chain_a.png'))
        style.plot_chain(chains[0],filename=os.path.join(plot_dir,'chain_all.png'))
        style.plot_gelman_rubin(chains,parameters=['parameters.a'],filename=os.path.join(plot_dir,'gr_a.png'))
        style.plot_gelman_rubin(chains,parameters=parameters,multivariate=True,filename=os.path.join(plot_dir,'gr_multivariate.png'))
        style.plot_autocorrelation_time(chains,parameters=['parameters.a'],threshold=50,filename=os.path.join(plot_dir,'autocorr_a.png'))

def test_misc():

    samples_dir = '_samples'
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    mean,cov,chains = get_chains(parameters,n=2)

    chain = chains[0]

    for chains in iterate_mpi(chains):

        chain = chains[0]
        chain['metrics.weight']
        assert chain.gget('parameters.a').size == chain.gsize
        chain.parameters['parameters.a'].latex = 'a'
        chain.parameters['parameters.a'].prior = Prior(limits=(-10.,10.))
        pb = chain.parameters['parameters.b']
        pb.prior = Prior(dist='norm',loc=1.,limits=(-10.,10.))
        pb = Parameter.from_state(pb.__getstate__())
        chain['metrics','logposterior'] = chain.zeros()
        fn = os.path.join(samples_dir,'samples.npy')
        chain.save(fn)
        chain1 = Samples.load(fn,mpistate='broadcast')
        base_fn = os.path.join(samples_dir,'samples')
        chain1.save_auto(base_fn + '.txt',ichain=0)
        chain1.save_cosmomc(base_fn,ichain=0)
        chain2 = Samples.load_cosmomc(base_fn,mpistate='scattered')
        chain1.mpi_scatter()
        for col in chain1.columns():
            if chain1.is_mpi_root():
                assert np.allclose(chain1[col],chain2[col])
        chain.to_getdist()
        assert np.allclose(chain.interval('parameters.a',method='exact'),chain.interval('parameters.a',method='gaussian_kde'),rtol=1e-1,atol=1e-1)
        chain3 = chain2.deepcopy()
        chain2['parameters.a'] += 1
        chain2.parameters['parameters.a'].latex = 'answer'
        if chain2.mpicomm.rank == 0:
            assert np.allclose(chain3['parameters.a'],chain2['parameters.a']-1)
        assert chain3.parameters['parameters.a'].latex != chain2.parameters['parameters.a'].latex
        gsize = chains[0].gsize + chains[1].gsize
        chains[0].extend(chains[1])
        assert chains[0].gsize == gsize
        chains[0].to_mesh('parameters.a')
        assert len(chains[0].interval('parameters.a')) == 2
        gsize = chain.gsize
        chain = chain.remove_burnin(0.2)
        assert chain.gsize < gsize
        assert chain == chain

    chain1 = chains[0]
    chain1.mpiroot = 0
    chain2 = chain1.deepcopy()
    if chain1.mpicomm.size > 1: chain2.mpiroot = 1
    chain1.mpi_scatter()
    chain2.mpi_scatter()
    assert chain1 == chain2


def test_stats():
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    mean,cov,chains = get_chains(parameters,n=4)

    try:
        from emcee import autocorr
        ref = autocorr.integrated_time(chains[0]['parameters.a'],quiet=True)
        assert np.allclose(chains[0].integrated_autocorrelation_time('parameters.a'),ref)
        tab = np.array([chain['parameters.a'] for chain in chains]).T
        ref = autocorr.integrated_time(tab,quiet=True)
        assert np.allclose(Samples.integrated_autocorrelation_time(chains,'parameters.a'),ref)
        assert np.allclose(Samples.integrated_autocorrelation_time(chains,['parameters.a']*2),ref[0])
    except ImportError:
        pass

    for chains in iterate_mpi(chains):
        assert np.allclose(Samples.gelman_rubin(chains,'parameters.a',method='diag'),Samples.gelman_rubin(chains,'parameters.a',method='eigen'))
        assert np.ndim(Samples.gelman_rubin(chains,'parameters.a',method='eigen')) == 0
        assert Samples.gelman_rubin(chains,['parameters.a'],method='eigen').shape == (1,)
        assert np.ndim(chains[0].integrated_autocorrelation_time('parameters.a')) == 0
        #print(chains[0].integrated_autocorrelation_time('parameters.a'))
        print(chains[0].to_stats(tablefmt='latex_raw'))
        #print(samples.to_stats(tablefmt='pretty'))


def test_mpi():
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    comm = CurrentMPIComm.get()
    samples = None
    size = 4000
    mean,cov,chains = get_chains(parameters,n=1,size=size)
    samples = chains[0]
    samples.mpi_scatter()
    samples = Samples.mpi_broadcast(samples)
    samples1 = samples.deepcopy()
    samples.mpi_scatter()
    assert samples.gsize == size
    samples.mpi_gather()
    samples2 = Samples.mpi_broadcast(samples)
    assert samples2 == samples1
    #print(samples.parameters)

    if comm.size > 3:
        color = comm.rank > 2
        newcomm = comm.Split(color, 0)
        samples = None
        mean,cov,chains = get_chains(parameters,n=1)
        samples = chains[0]
        if color:
            #samples = Samples.mpi_broadcast(samples,mpicomm=newcomm)
            #mean,cov,chains = get_chains(parameters,n=1)
            #samples = chains[0]
            samples.mpicomm = newcomm
            samples.mpistate = 'scattered'

        samples = Samples.mpi_collect(samples,sources=range(3,comm.size),mpicomm=comm)
        comm.Barrier()

        mean,cov,chains = get_chains(parameters,n=1)
        samples = chains[0]
        samples.mpicomm = comm
        samples.mpistate = 'scattered'

        samples2 = samples.mpi_distribute(dests=range(3,comm.size),mpicomm=newcomm)
        if comm.rank in range(3,comm.size):
            print(samples2.size,samples2.gsize,samples2['parameters.a'])


if __name__ == '__main__':

    setup_logging()
    test_plotting()
    test_misc()
    test_stats()
    test_mpi()
