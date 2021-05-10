import numpy as np


def match1d(id1,id2):
    """
    Match id2 to id1.

    Parameters
    ----------
    id1 : array-like
        IDs 1, should be unique.

    id2 : array-like
        IDs 2, should be unique.

    Returns
    -------
    index1 : ndarray
        Indices of matching ``id1``.

    index2 : ndarray
        Indices of matching ``id2``.

    Warning
    -------
    Makes sense only if ``id1`` and ``id2`` elements are unique.

    References
    ----------
    https://www.followthesheep.com/?p=1366
    """
    sort1 = np.argsort(id1)
    sort2 = np.argsort(id2)
    sortleft1 = id1[sort1].searchsorted(id2[sort2],side='left')
    sortright1 = id1[sort1].searchsorted(id2[sort2],side='right')
    sortleft2 = id2[sort2].searchsorted(id1[sort1],side='left')
    sortright2 = id2[sort2].searchsorted(id1[sort1],side='right')

    ind2 = np.flatnonzero(sortright1-sortleft1 > 0)
    ind1 = np.flatnonzero(sortright2-sortleft2 > 0)

    return sort1[ind1], sort2[ind2]
