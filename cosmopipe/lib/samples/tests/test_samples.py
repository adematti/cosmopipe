import os

import numpy as np
from cosmopipe.lib.mpi import CurrentMPIComm

from cosmopipe.lib import setup_logging
from cosmopipe.lib.parameter import Parameter, ParamName, Prior
from cosmopipe.lib.samples import Samples, SamplesPlotStyle
from cosmopipe.lib.samples.plotting import plot_normal_1d, plot_normal_2d


def get_chains(parameters, n=4):
    rng = np.random.RandomState()
    ndim = len(parameters)
    mean = np.zeros(ndim,dtype='f8')
    cov = np.eye(ndim,dtype='f8')
    cov += 0.1 # off-diagonal
    invcov = np.linalg.inv(cov)
    chains = []
    for ichain in range(4):
        array = rng.multivariate_normal(mean,cov,size=4000)
        diff = array-mean
        samples = Samples.from_array(array.T,columns=parameters)
        samples['metrics','logposterior'] = -1./2.*np.sum(diff.dot(invcov)*diff,axis=-1)
        for par in parameters:
            samples.parameters[par].fixed = False
        chains.append(samples)
    return mean,cov,chains


def test_plotting():

    plot_dir = '_plots'
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    mean,cov,chains = get_chains(parameters,n=4)
    """
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
    """
    style = SamplesPlotStyle()
    style.plot_chain(chains[0],parameters=['parameters.a'],filename=os.path.join(plot_dir,'chain_a.png'))
    style.plot_chain(chains[0],filename=os.path.join(plot_dir,'chain_all.png'))
    style.plot_gelman_rubin(chains,parameters=['parameters.a'],filename=os.path.join(plot_dir,'gr_a.png'))
    style.plot_gelman_rubin(chains,parameters=parameters,multivariate=True,filename=os.path.join(plot_dir,'gr_multivariate.png'))
    style.plot_autocorrelation_time(chains,parameters=['parameters.a'],threshold=50,filename=os.path.join(plot_dir,'autocorr_a.png'))


def test_misc():

    samples_dir = '_samples'
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    mean,cov,chains = get_chains(parameters,n=4)
    chain = chains[0]
    chain.add_default_weight()
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
    chain1.save_cosmomc(base_fn,ichain=0)
    chain2 = Samples.load_auto(base_fn + '.txt',mpistate='broadcast')
    chain2 = Samples.load_cosmomc(base_fn,mpistate='scattered')
    chain.mpi_scatter()
    assert set(chain2.columns()) == set(chain.columns())
    if chain.mpicomm.rank == 0:
        for col in chain.columns():
            assert np.allclose(chain[col],chain2[col])
    chain.to_getdist()
    assert np.allclose(chain.interval('parameters.a',method='exact'),chain.interval('parameters.a',method='gaussian_kde'),rtol=1e-1,atol=1e-1)
    chain3 = chain2.deepcopy()
    chain2['parameters.a'] += 1
    chain2.parameters['parameters.a'].latex = 'answer'
    if chain2.mpicomm.rank == 0:
        assert np.allclose(chain3['parameters.a'],chain2['parameters.a']-1)
    assert chain3.parameters['parameters.a'].latex != chain2.parameters['parameters.a'].latex


def test_stats():

    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    mean,cov,chains = get_chains(parameters,n=1)
    samples = chains[0]
    samples.mpi_scatter()
    print(samples.to_stats(tablefmt='latex_raw'))
    print(samples.to_stats(tablefmt='pretty'))


def test_mpi():
    """
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    comm = CurrentMPIComm.get()
    samples = None
    if comm.rank == 0:
        mean,cov,chains = get_chains(parameters,n=1)
        samples = chains[0]
        samples.add_default_weight()
    samples = Samples.mpi_broadcast(samples)
    samples1 = samples.deepcopy()
    samples.mpi_scatter()
    assert samples.gsize == 4000
    samples.mpi_gather()
    samples2 = Samples.mpi_broadcast(samples)
    assert samples2 == samples1
    #print(samples.parameters)
    """
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    comm = CurrentMPIComm.get()
    mean,cov,chains = get_chains(parameters,n=1)
    samples = chains[0]
    samples.mpistate = 'scattered'
    gsize = samples.gsize
    samples = samples.remove_burnin(0.2)
    assert samples.gsize < gsize

    color = comm.rank > 2
    newcomm = comm.Split(color, 0)
    samples = None
    if newcomm.rank == 0:
        mean,cov,chains = get_chains(parameters,n=1)
        samples = chains[0]
        samples.add_default_weight()
    if color:
        #samples = Samples.mpi_broadcast(samples,mpicomm=newcomm)
        mean,cov,chains = get_chains(parameters,n=1)
        samples = chains[0]
        samples.mpicomm = newcomm
        samples.mpistate = 'scattered'
        samples.add_default_weight()
        #print(samples['parameters.a'])
        #samples.mpi_gather()
        #samples = Samples.mpi_broadcast(samples,mpicomm=newcomm)
        #samples.mpi_send(newcomm=comm,tag=42)
    #newcomm.Barrier()
    #comm.Barrier()
    #if comm.rank == 0: print('======================================================================')
    samples = Samples.mpi_collect(samples,sources=range(3,comm.size),mpicomm=comm)
    #print(samples.parameters)
    #print(samples['parameters.a'])
    #exit()
    #print(samples.columns(),samples.size,samples.gsize)
    #exit()
    comm.Barrier()
    #print(samples.mpicomm.size)

    mean,cov,chains = get_chains(parameters,n=1)
    samples = chains[0]
    samples.mpicomm = comm
    samples.mpistate = 'scattered'
    samples.add_default_weight()

    samples2 = samples.mpi_distribute(dests=range(3,comm.size),mpicomm=newcomm)
    #print(comm.rank,samples2.mpicomm.size,samples2.mpiroot,samples2.mpistate)
    if comm.rank in range(3,comm.size):
        print(samples2.size,samples2.gsize,samples2['parameters.a'])
    #print(comm.rank,samples.columns(),samples.size,samples.gsize)



if __name__ == '__main__':

    setup_logging()
    #test_plotting()
    test_misc()
    #test_mpi()
    #test_stats()
