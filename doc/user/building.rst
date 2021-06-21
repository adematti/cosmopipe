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
  pip install git+https://github.com/adematti/pypescript

If you want to use **nbodykit** (for e.g. power spectrum estimation) as well, it is easier to install through conda (pip would recompile from source)::

  conda install nbodykit

and eventually **cosmopipe**::

  pip install git+https://github.com/adematti/cosmopipe

.. note::

  At NERSC, you may have to unload desiconda module first

.. note::

  The list of modules (and hence the packages they rely on) to install is given by :root:`install_modules.txt`,
  using Unix filename pattern matching.
  If you want to change it, clone the **cosmopipe** github repository, modify this list, and pip install **cosmopipe**::

    git clone https://github.com/adematti/cosmopipe
    vi install_modules.txt # write changes; module one wants to exclude start with "!"
    python -m pip install .

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

Following `JupyterHub with shifter`_, JupyterHub can use your shifter image as kernel. Simply create a file: ``~/.local/share/jupyter/kernels/cosmopipe-shifter/kernel.json``
with the following content::

  {
    "argv": [
        "shifter",
        "--image={dockerimage}",
        "/opt/conda/bin/python",
        "-m",
        "ipykernel_launcher",
        "-f",
        "{connection_file}"
    ],
    "display_name": "my-shifter-kernel",
    "language": "python"
  }

.. note::

  For further information on shifter, see `shifter docs`_.

References
==========

.. target-notes::

.. _`pypescript`: https://github.com/adematti/pypescript

.. _`shifter docs`: https://shifter.readthedocs.io

.. _`JupyterHub with shifter`: https://docs.nersc.gov/services/jupyter/#shifter-kernels-on-jupyter
