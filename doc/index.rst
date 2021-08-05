.. title:: cosmopipe docs

*************************************
Welcome to cosmopipe's documentation!
*************************************

.. toctree::
  :maxdepth: 1
  :caption: User documentation

  user/building
  api/api
  api/modules

.. toctree::
  :maxdepth: 1
  :caption: Developer documentation

  developer/documentation
  developer/tests
  developer/contributing
  developer/changes

.. toctree::
  :hidden:

************
Introduction
************

**cosmopipe** is a framework based on **pypescript** to extract cosmological constraints from galaxy redshift surveys.
This is **not** a cosmological inference code, such a `Cobaya`_, `MontePython`_, `CosmoMC`_ at al.,
which are designed to sample pre-coded or user-provided likelihoods.
Rather, **cosmopipe** provides coherent interfaces (called *modules*) to different codes (e.g. power spectrum estimator,
primordial perturbations, perturbation theory code, geometry effects, samplers), that handle common data types (data vectors, samples, etc.).
`pypescript`_ is then used to articulate these modules into analysis pipelines provided a configuration file.
Running a clustering analysis is then as simple as::

  pypescript myconfig.yaml

(probably using MPI.)

The rationale
=============

As we are entering the era of precision cosmology with spectroscopic surveys, it becomes mandatory to provide an *easy* way for our collaborators
to reproduce our analyses, from (at least) the clustering catalogs down to the cosmological constraints.
Our collaborators should not have to clone our codes (e.g. theory models) from all over github; these codes should be linked in one place
and run with the correct versions.
This would be possible by running the codes inside the safe, controlled environment of a `Docker`_ container.

Our collaborators should not have to worry about having the correct parameter files for each code, nor how to connect the output of one code
to the input of another. Ideally, they should only have to deal with *one* (provided) parameter file and *one* output,
and should not need to know anything about cosmology in general to rerun our analysis.
This would be possible by linking all our codes together to form a *pipeline*.

This *pipeline* could be made flexible enough to cover the clustering analyses of the collaboration. For this, we should not have *one* pipeline,
but a framework to *script* pipelines. For example, once codes (hereafter *modules*) to model 2 and 3-pt correlation functions
(as well as their covariance) are available, it should be straightforward for anyone to run a joint 2 and 3-pt analysis, without coding anything else.

Eventually, it should be made *very* easy to code and include new modules, without the need for a global view of the pipeline.
This would be possible by providing a simple module architecture to copy and fill, with either Python, C, C++ or Fortran code.

These should be the key features of **cosmopipe**, quite similar to `CosmoSIS`_ used by the DES collaboration.

Acknowledgements
================

No acknowledgements to be made yet!

Changelog
=========

* :doc:`developer/changes`

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

References
==========

.. target-notes::

.. _`pypescript`: https://github.com/adematti/pypescript

.. _`Cobaya`: https://github.com/CobayaSampler/cobaya

.. _`MontePython`: https://github.com/baudren/montepython_public

.. _`CosmoMC`: https://github.com/cmbant/CosmoMC

.. _`Docker`: https://www.docker.com/

.. _`CosmoSIS`: https://bitbucket.org/joezuntz/cosmosis/src/master
