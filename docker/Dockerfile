FROM ubuntu:18.04

SHELL ["/bin/bash", "-c"]

RUN \
    apt-get update && \
    apt-get install --yes \
        build-essential \
        gfortran \
        python3-dev \
        wget \
        vim \
        git \
        libgsl-dev \
        libbz2-dev \
        libcfitsio-dev \
        libcfitsio-bin \
        curl \
        patch \
        ca-certificates && \
    apt-get clean all

ARG mpich=3.3
ARG mpich_prefix=mpich-$mpich

RUN \
    wget https://www.mpich.org/static/downloads/$mpich/$mpich_prefix.tar.gz && \
    tar xvzf $mpich_prefix.tar.gz                                           && \
    cd $mpich_prefix                                                        && \
    ./configure                                                             && \
    make -j 4                                                               && \
    make install                                                            && \
    make clean                                                              && \
    cd ..                                                                   && \
    rm -rf $mpich_prefix

ARG mpi4py=3.0.3
ARG mpi4py_prefix=mpi4py-$mpi4py

RUN \
    wget https://bitbucket.org/mpi4py/mpi4py/downloads/$mpi4py_prefix.tar.gz && \
    tar xvzf $mpi4py_prefix.tar.gz                                           && \
    cd $mpi4py_prefix                                                        && \
    python3 setup.py build                                                   && \
    python3 setup.py install                                                 && \
    cd ..                                                                    && \
    rm -rf $mpi4py_prefix


RUN wget -nv https://repo.continuum.io/miniconda/Miniconda3-4.5.1-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda

RUN ln -s /opt/conda/bin/conda /bin/conda && \
    conda config --add channels intel && \
    conda config --set always_yes yes && \
    conda update conda && \
    conda init bash

RUN chmod 777 /opt/conda /root

RUN conda install pip numpy pyyaml cython sphinx_rtd_theme

RUN conda install -c bccp nbodykit

RUN ln -s /opt/conda/bin/pip /bin/pip

ARG COSMOPIPE_VER=unknown

RUN for x in \
  git+https://github.com/adematti/pypescript \
  git+https://github.com/adematti/cosmopipe \
  ; do pip install $x; done \
  && rm -Rf /root/.cache/pip

WORKDIR /src/

RUN git clone http://bitbucket.org/joezuntz/cosmosis \
  && cd cosmosis \
  && git checkout \
  && git clone http://bitbucket.org/joezuntz/cosmosis-standard-library

#RUN cd cosmosis \
#  && source config/setup-cosmosis \
#  && make

#ENV COSMOSIS_SRC_DIR=

ENV MPLBACKEND Agg

# COPY . /src/cosmopipe

# ENV PYTHONPATH /src/cosmopipe:${PYTHONPATH}

WORKDIR /homedir/

RUN /sbin/ldconfig
