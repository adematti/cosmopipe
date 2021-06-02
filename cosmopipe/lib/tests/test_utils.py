import numpy as np

from cosmopipe.lib import setup_logging, utils

def intbin_ndarray(ndarray, new_shape, weights=None, operation=np.sum):
    """Bin an ndarray in all axes based on the target shape, by summing or
    averaging. Number of output dimensions must match number of input dimensions and
    new axes must divide old ones.

    Taken from https://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    and https://nbodykit.readthedocs.io/en/latest/_modules/nbodykit/binned_statistic.html#BinnedStatistic.reindex.

    Example
    -------
    >>> m = np.arange(0,100,1).reshape((10,10))
    >>> n = bin_ndarray(m, new_shape=(5,5), operation='sum')
    >>> print(n)

    [[ 22  30  38  46  54]
     [102 110 118 126 134]
     [182 190 198 206 214]
     [262 270 278 286 294]
     [342 350 358 366 374]]

    """
    if ndarray.ndim != len(new_shape):
        raise ValueError('Shape mismatch: {} -> {}'.format(ndarray.shape,new_shape))
    if any([c % d != 0 for d,c in zip(new_shape,ndarray.shape)]):
        raise ValueError('New shape must be a divider of the original shape')
    compression_pairs = [(d, c//d) for d,c in zip(new_shape,ndarray.shape)]
    flattened = [l for p in compression_pairs for l in p]
    ndarray = ndarray.reshape(flattened)
    if weights is not None: weights = weights.reshape(flattened)

    for i in range(len(new_shape)):
        if weights is not None:
            ndarray = operation(ndarray, weights=weights, axis=-1*(i+1))
        else:
            ndarray = operation(ndarray, axis=-1*(i+1))

    return ndarray


def test_misc():
    a = 2
    assert utils.cov_to_corrcoef(a) == 1.
    a = np.arange(18).reshape((3,6))
    ref = intbin_ndarray(a,(3,3))
    res = utils.rebin_ndarray(a,(3,3),interpolation=False)
    assert np.allclose(ref,res)
    res = utils.rebin_ndarray(a,(3,3))
    assert np.allclose(ref,res)
    ref = utils.rebin_ndarray(a,(3,np.arange(5)))
    res = utils.rebin_ndarray(a,(3,np.arange(5)-0.1),interpolation=False)
    assert np.allclose(ref,res)



def test_round():
    assert utils.round_measurement(0.01,-1.0,1.0,sigfigs=2) == ('0.0', '-1.0', '1.0')
    assert utils.round_measurement(0.01,-1.0,0.8,sigfigs=2) == ('0.01', '-1.00', '0.80')
    assert utils.round_measurement(0.0001,-1.0,1.0,sigfigs=2)  == ('0.0', '-1.0', '1.0')
    assert utils.round_measurement(1e4,-1.0,1.0,sigfigs=2) == ('1.00000e4', '-1.0e0', '1.0e0')
    assert utils.round_measurement(-0.0001,-1.0,1.0,sigfigs=2) == ('0.0', '-1.0', '1.0')


if __name__ == '__main__':

    setup_logging()
    test_misc()
    test_round()
