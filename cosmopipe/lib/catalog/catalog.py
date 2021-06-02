import logging

from .base import BaseCatalog


def _multiple_columns(column):
    return isinstance(column,(list,ParamBlock))


def vectorize_columns(func):
    @functools.wraps(func)
    def wrapper(self, column, **kwargs):
        if not _multiple_columns(column):
            return func(self,column,**kwargs)
        toret = [func(self,col,**kwargs) for col in column]
        if all(t is None for t in toret): # in case not broadcast to all ranks
            return None
        return np.asarray(toret)
    return wrapper


class Catalog(BaseCatalog):

    logger = logging.getLogger('Catalog')
