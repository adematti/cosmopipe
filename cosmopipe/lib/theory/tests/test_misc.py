import numpy as np
from scipy import interpolate, special

from cosmopipe.lib.utils import MemoryMonitor


def test_interp():

    k = np.linspace(0,1,100)
    pk = np.linspace(0,1,100)
    kint = k[:-1]+0.0042

    num = int(1e5)
    with MemoryMonitor(other='numpy_interp') as mem:
        for i in range(num):
            np.interp(kint,k,pk)

    with MemoryMonitor(other='scipy_interp') as mem: # 20x slower
        for i in range(num):
            interpolate.interp1d(k,pk,kind='linear',axis=-1,copy=False,bounds_error=True,assume_sorted=True)(kint)


def test_legendre():
    mu = np.linspace(0,1,100)
    num = int(1e5)
    with MemoryMonitor(other='full') as mem:
        for i in range(num):
            special.legendre(2)(mu)
    leg = special.legendre(2)
    with MemoryMonitor(other='init') as mem:
        for i in range(num):
            leg(mu)


if __name__ == '__main__':

    #test_interp()
    test_legendre()
