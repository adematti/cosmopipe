import os
import numpy as np

from cosmopipe.lib import setup_logging
from cosmopipe.lib.samples import Profiles, ProfilesPlotStyle


def get_profiles(parameters):
    size = 10
    rng = np.random.RandomState()
    ndim = len(parameters)
    mean = np.zeros(ndim,dtype='f8')
    std = np.ones(ndim,dtype='f8')
    profiles = Profiles(parameters=parameters)
    profiles.set_bestfit({param:rng.normal(loc=mean_,scale=std_,size=size) for param,mean_,std_ in zip(parameters,mean,std)})
    profiles.set_parabolic_errors({param:rng.normal(loc=std_,scale=std_/10.,size=size) for param,mean_,std_ in zip(parameters,mean,std)})
    profiles.set_deltachi2_errors({param:np.array([-rng.normal(loc=std_,scale=std_/10.,size=size),rng.normal(loc=std_,scale=std_,size=size)]).T\
                                    for param,mean_,std_ in zip(parameters,mean,std)})
    profiles.set_metrics({'minchi2':rng.normal(loc=10,scale=0.1,size=size)})
    return profiles


def test_stats():
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    profiles = get_profiles(parameters)
    print(profiles.to_stats(tablefmt='latex_raw'))
    print(profiles.to_stats(tablefmt='pretty'))


def test_plotting():
    plot_dir = '_plots'
    parameters = ['parameters.a','parameters.b','parameters.c','parameters.d']
    profiles = get_profiles(parameters)
    assert 'parameters.a' in profiles.bestfit
    style = ProfilesPlotStyle()
    style.plot_aligned(profiles,'parameters.a',truth=0.,yband=(-1.,1.,'abs'),filename=os.path.join(plot_dir,'aligned_a.png'))
    profiles = [get_profiles(parameters) for i in range(100)]
    style.plot_aligned_stacked(profiles[:4],['parameters.a','parameters.b'],truths=[0.]*2,ybands=[(-1.,1.,'abs')]*2,filename=os.path.join(plot_dir,'aligned_stacked_a.png'))
    style.plot_1d(profiles,parameter='parameters.a',filename=os.path.join(plot_dir,'kstest_a.png'))
    style.plot_2d(profiles,parameters=['parameters.a','parameters.b'],filename=os.path.join(plot_dir,'scatter_ab.png'))
    style.plot_corner(profiles,filename=os.path.join(plot_dir,'corner.png'))

if __name__ == '__main__':

    setup_logging()
    test_stats()
    test_plotting()
