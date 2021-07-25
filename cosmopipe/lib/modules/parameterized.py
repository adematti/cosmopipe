from pypescript import BaseModule

from cosmopipe.lib.parameter import ParamBlock, ParamError, find_names
from cosmopipe.lib import syntax
from cosmopipe import section_names


class ParameterizedModule(BaseModule):

    def set_param_block(self, include=None):
        base = self.options.get('base_parameters',{})
        extra = self.options.get('update_parameters',{}) or {}
        base = ParamBlock(syntax.collapse_sections(base,maxdepth=2))
        if include is not None:
            for param in base:
                if param.name not in include:
                    del base[param]
        fix = extra.pop('fix',[])
        extra = syntax.collapse_sections(extra,maxdepth=2)
        datablock_mapping = {}
        allnames = list(map(str,base.names()))
        for name,update in extra.items():
            update = update.copy() # copy to avoid modifying config...
            specific = update.pop('specific',False)
            set_latex = 'latex' in update
            latex = update.pop('latex',None)
            names = find_names(allnames,name,latex=latex)
            if names is None:
                raise ParamError('Parameter {} not found in {}'.format(name,allnames))
            for name,latex in names:
                param = base[name]
                if set_latex:
                    param.update(latex=latex,**update)
                else:
                    param.update(**update)
                if specific:
                    param_name = param.name.copy()
                    param.add_suffix(self.name if isinstance(specific,bool) else specific)
                    datablock_mapping[param_name.tuple] = param.name.tuple
        for name in fix:
            base[name].fixed = True
        for param in base:
            if param.value is None:
                raise ParamError('An initial value must be provided for parameter {}.'.format(param.name))
        for param in base:
            self.data_block.setdefault(*param.name.tuple,param.value)
        paramblock = self.data_block.get(section_names.parameters,'list',[])
        paramblock += base
        self.data_block[section_names.parameters,'list'] = paramblock
        #for param in self.data_block[section_names.parameters,'list']:
        #    print(repr(param))
        self._datablock_mapping.update(datablock_mapping)
