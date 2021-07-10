# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

#example how to script input parameter changes
#(of course could construct the parameter dictionary from scratch here)
import pypescript.main
import pypescript.config
from cosmopipe.theory.galaxy_clustering.velocileptors import get_bias_parameters

params=pypescript.config.ConfigBlock("demo_velocileptors_parameterhacking.yaml")

biases=get_bias_parameters()

for key in biases['galaxy_bias'] :
    if key != 'b1' : biases['galaxy_bias'][key]['fixed']=True

params.data['params']['common_parameters'].update(biases)

params.data['mockdata']['mean']=False

pypescript.main.main(params)


