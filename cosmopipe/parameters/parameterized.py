from pypescript import BaseModule

from cosmopipe.lib.parameter import ParamError, find_names, find_names_latex
from cosmopipe.lib import syntax
from cosmopipe import section_names


class ParameterizedModule(BaseModule):

    def set_parameters(self, include=None, exclude=None):
        base = self.options.get('base_parameters',{})
        extra = self.options.get('update_parameters',{}) or {}
        from cosmopipe.lib.parameter import ParameterCollection
        base = ParameterCollection(syntax.collapse_sections(base,maxdepth=2))
        extra = syntax.collapse_sections(extra,maxdepth=2).copy()
        fixed = extra.pop('fixed',[])
        varied = extra.pop('varied',[])
        derived = extra.pop('derived',[])
        allnames = list(map(str,base.names()))
        if include is not None:
            for name in find_names(allnames,include,quiet=False):
                del base[name]
        exclude = (exclude or []) + derived
        self._derived_parameters = ParameterCollection()
        for name in find_names(allnames,exclude,quiet=False):
            self._derived_parameters.set(base[name])
            del base[name]
        derived = [str(name) for name in self.data_block.get(section_names.parameters,'derived',[])]
        for name in find_names(allnames,derived,quiet=True):
            self._derived_parameters.set(base[name])
            del base[name]
        datablock_mapping = {}
        allnames = list(map(str,base.names()))
        for name,update in extra.items():
            update = update.copy() # copy to avoid modifying config...
            specific = update.pop('specific',False)
            set_latex = 'latex' in update
            latex = update.pop('latex',None)
            if set_latex:
                names = find_names_latex(allnames,name,latex=latex,quiet=False)
            else:
                names = find_names(allnames,name,quiet=False)
            if not names:
                raise ParamError('Parameter {} not found in {}'.format(name,allnames))
            for name in names:
                if set_latex:
                    name,latex = name
                    param = base[name]
                    param.update(latex=latex,**update)
                else:
                    param = base[name]
                    param.update(**update)
                if specific:
                    param_name = param.name.copy()
                    param.add_suffix(self.name if isinstance(specific,bool) else specific)
                    datablock_mapping[param_name.tuple] = param.name.tuple
        for name in find_names(allnames,fixed,quiet=False):
            base[name].fixed = True
        for name in find_names(allnames,varied,quiet=False):
            base[name].fixed = False
        for param in base:
            if param.value is None:
                raise ParamError('An initial value must be provided for parameter {}.'.format(param.name))
        for param in base:
            self.data_block.setdefault(*param.name.tuple,param.value)
        ParameterCollection = self.data_block.get(section_names.parameters,'list',[])
        ParameterCollection += base
        self.data_block[section_names.parameters,'list'] = ParameterCollection
        #for param in self.data_block[section_names.parameters,'list']:
        #    print(repr(param))
        self._datablock_mapping.update(datablock_mapping)
