.. _user-building:

Building
========

**cosmopipe** relies on the `pypescript`_ framework.

Conda
-----

First create a conda environment::

  conda create -n cosmopipe
  conda activate cosmopipe

Then install some modules required for the installation of **cosmopipe** (and dependencies)::

  conda install numpy mpi4py pyyaml cython
  conda install -c bccp nbodykit
  pip install git+https://github.com/adematti/pypescript

and eventually **cosmopipe**, with all modules::

  pip install git+https://github.com/adematti/cosmopipe#egg=cosmopipe[all]

.. note::

  At NERSC, you may have to unload desiconda module first if you encounter trouble in the installation of **pypescript**


.. note::

  When **cosmopipe** builds with the option [all], it fetches all the modules to be installed, list their dependencies
  which **pip** then takes care to install.
  The list of modules (and hence the packages they rely on) to install is given by :root:`install_modules.txt`,
  using Unix filename pattern matching.
  If you want to change it, clone the **cosmopipe** github repository, modify this list, and pip install **cosmopipe**::

    git clone https://github.com/adematti/cosmopipe
    cd cosmopipe
    vi install_modules.txt # write changes; module one wants to exclude start with "!"
    python -m pip install .[all]

.. note::

  If you do not need to use modules for estimation of power spectra or correlation functions, you can add in install_modules.txt::

    !estimators.*

  and ignore installation of nbodykit (conda install -c bccp nbodykit).


At NERSC
--------

At NERSC, if you want to use **cosmopipe** on Cori nodes, **mpi4py** should *not* be installed using conda-provided prebuilt distribution, see `parallel python`_.

Current solution is to clone NERSC's **nbodykit** environment::

  source /global/common/software/m3035/conda-activate.sh 3.7
  conda create -n cosmopipe --clone bcast-bccp-3.7
  salloc -C haswell -t 00:20:00 --qos interactive -L SCRATCH,project
  MPICC="cc -shared" pip install git+https://github.com/adematti/pypescript
  pip install git+https://github.com/adematti/cosmopipe


Docker
------

**cosmopipe** can be run from a laptop and NERSC using the same Docker image.
The Docker image is available on :dockerroot:`Docker Hub <>`.

On your laptop
^^^^^^^^^^^^^^
First pull::

  docker pull {dockerimage}

To run on-the-fly::

  docker run {dockerimage} pypescript myconfig.yaml

Or in interactive mode, you can bind mount your working directory ``absolutepath``::

  docker run --volume absolutepath:/homedir/ -it {dockerimage} /bin/bash

which allows you to work as usual (type ``exit`` to exit).


At NERSC
^^^^^^^^
First pull::

  shifterimg -v pull {dockerimage}

To run on-the-fly::

  shifter --module=mpich-cle6 --image={dockerimage} pypescript myconfig.yaml

In interactive mode::

  shifter --volume absolutepath:/homedir/ --image={dockerimage} /bin/bash

.. note::

  For further information on shifter, see `shifter docs`_.

References
----------

.. target-notes::

.. _`pypescript`: https://github.com/adematti/pypescript

.. _`parallel python`: https://docs.nersc.gov/development/languages/python/parallel-python/

.. _`shifter docs`: https://shifter.readthedocs.io
