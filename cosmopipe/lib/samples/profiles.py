import logging

import numpy as np
import tabulate

from cosmopipe.lib.utils import BaseClass
from cosmopipe.lib import utils
from cosmopipe.lib.parameter import ParamBlock, Parameter, ParamName


class ParamDict(BaseClass):

    def __init__(self, data):
        self.data = {}
        for key,value in data.items():
            self[key] = value

    def columns(self):
        return list(self.data.keys())

    def __getitem__(self, name):
        if isinstance(name,(Parameter,ParamName,str,tuple)):
            return self.data[ParamName(name)]
        else:
            return self.__class__({col:self[col][name] for col in self.columns()})

    def __setitem__(self, name, item):
        if isinstance(name,(Parameter,ParamName,str,tuple)):
            self.data[ParamName(name)] = item
        else:
            for col in self.columns():
                self[col][name] = item

    def __getstate__(self):
        """Return this class state dictionary."""
        data = {str(name):col for name,col in self.data.items()}
        return {'data':data}

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.data = {ParamName(name):col for name,col in state['data'].items()}

    def items(self):
        return self.data.items()

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,self.data)

    def __len__(self):
        return len(self.data[self.columns()[0]])

    def __contains__(self, name):
        return ParamName(name) in self.data


class ParamCovariance(BaseClass):

    def __init__(self, mat, parameters):
        self.mat = mat
        self.parameters = parameters

    def cov(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        idx = np.array([self.parameters.index(param) for param in parameters])
        toret = self.mat[np.ix_(idx,idx)]
        return toret

    def invcov(self, parameters=None, **kwargs):
        return utils.inv(self.cov(parameters,**kwargs))

    def corrcoef(self, parameters=None):
        return utils.cov_to_corrcoef(self.cov(parmeters=parameters))

    def __getstate__(self):
        """Return this class state dictionary."""
        return {'mat':self.mat,'parameters':self.parameters.__getstate__()}

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.mat = state['mat']
        self.parameters = ParamBlock.from_state(state['parameters'])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,self.mat)


class Profiles(BaseClass):

    logger = logging.getLogger('Profiles')
    _paramdicts = ['metrics','init','bestfit','parabolic_errors','deltachi2_errors']

    def __init__(self, parameters=None, attrs=None):
        self.parameters = ParamBlock(parameters)
        self.attrs = attrs or {}

    def __len__(self):
        return len(self.bestfit)

    def set_metrics(self, values):
        self.metrics = ParamDict(values)

    def set_init(self, values):
        self.init = ParamDict(values)

    def set_bestfit(self, values):
        self.bestfit = ParamDict(values)

    def set_parabolic_errors(self, values):
        self.parabolic_errors = ParamDict(values)

    def set_deltachi2_errors(self, values):
        self.deltachi2_errors = ParamDict(values)

    def set_covariance(self, mat, parameters=None):
        self.covariance = ParamCovariance(mat,parameters=parameters if parameters is not None else self.parameters)

    def argmin(self):
        return self.metrics['minchi2'].argmin()

    def ntries(self):
        return len(self.metrics['minchi2'])

    def get(self, name):
        return getattr(self,name)

    def append(self, name):
        sntries = self.ntries()
        for name in ['init','metrics','bestfit','parabolic_errors']:
            if hasattr(self,name) and hasattr(other,name):
                setattr(self,name,np.concatenate([getattr(self,name),getattr(other,name)],axis=0))
            elif hasattr(self,name) ^ hasattr(other,name):
                raise ValueError('Cannot append two profiles if both do not have {}.'.format(name))
        for name in ['covariance','deltachi2_errors']:
            if self.argmin() >= sntries:
                if not hasattr(other,name):
                    raise ValueError('{} not provided for the global bestfit.'.format(name))
                setattr(self,name,getattr(other,name).copy())

    def __getstate__(self):
        """Return this class state dictionary."""
        state = {}
        state['attrs'] = self.attrs
        for name in ['parameters'] + self._paramdicts + ['covariance']:
            if hasattr(self,name):
                state[name] = getattr(self,name).__getstate__()
        return state

    def __setstate__(self, state):
        """Set the class state dictionary."""
        self.parameters = ParamBlock.from_state(state['parameters'])
        self.attrs = state['attrs']
        for name in self._paramdicts:
            if name in state:
                setattr(self,name,ParamDict.from_state(state[name]))
        if 'covariance' in state:
            self.covariance = ParamCovariance.from_state(state['covariance'])

    def __repr__(self):
        return '{}({})'.format(self.__class__.__name__,self.parameters)

    @classmethod
    def to_samples(cls, profiles, parameters=None, name='bestfit', select='best'):
        from .samples import Samples
        if parameters is None: parameters = profiles[0].parameters
        parameters = [profiles[0].parameters[param] for param in parameters]
        samples = Samples(parameters=parameters,mpicomm=profiles[0].mpicomm)
        for param in parameters:
            if select == 'best':
                samples[param] = np.array([prof.get(name)[param][prof.argmin()] for prof in profiles])
                samples['metrics.logposterior'] = -0.5*np.array([prof.metrics['minchi2'][prof.argmin()] for prof in profiles])
        return samples


    def to_stats(self, parameters=None, quantities=None, sigfigs=2, tablefmt='latex_raw', filename=None):
        if parameters is None: parameters = self.parameters
        data = []
        if quantities is None: quantities = ['bestfit','parabolic_errors','deltachi2_errors']
        is_latex = 'latex_raw' in tablefmt
        argmin = self.argmin()

        def round_errors(low, up):
            low,up = utils.round_measurement(0.0,low,up,sigfigs=sigfigs)[1:]
            if is_latex: return '${{}}_{{{}}}^{{+{}}}$'.format(low,up)
            return '{}/+{}'.format(low,up)

        for iparam,param in enumerate(parameters):
            row = []
            if is_latex: row.append(param.get_label())
            else: row.append(str(param.name))
            ref_error = self.parabolic_errors[param][argmin]
            for quantity in quantities:
                if quantity in ['bestfit','parabolic_errors']:
                    value = self.get(quantity)[param][argmin]
                    value = utils.round_measurement(value,ref_error,sigfigs=sigfigs)[0]
                    if is_latex: value = '${}$'.format(value)
                    row.append(value)
                elif quantity == 'deltachi2_errors':
                    low,up = self.get(quantity)[param][argmin]
                    row.append(round_errors(low,up))
                else:
                    raise RuntimeError('Unknown quantity {}.'.format(quantity))
            data.append(row)
        chi2min = '{:.2f}'.format(self.metrics['minchi2'][argmin])
        headers = [('$\chi^{{2}} = {}$' if is_latex else 'chi2 = {}').format(chi2min)]
        headers += [quantity.replace('_',' ') for quantity in quantities]
        tab = tabulate.tabulate(data,headers=headers,tablefmt=tablefmt)
        if filename and self.is_mpi_root():
            utils.mkdir(os.path.dirname(filename))
            self.log_info('Saving to {}.'.format(filename),rank=0)
            with open(filename,'w') as file:
                file.write(tab)
        return tab
