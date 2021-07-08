FROM ubuntu:20.04

SHELL ["/bin/bash", "-c"]

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && \
    apt-get install --yes --no-install-recommends \
        build-essential \
        gfortran \
        python3-dev \
        python3-pip \
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

RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install numpy pyyaml cython fitsio sphinx_rtd_theme

ARG COSMOPIPE_VER=unknown

RUN for x in \
  git+https://github.com/adematti/pypescript \
  ; do pip install $x; done \
  && rm -Rf /root/.cache/pip

WORKDIR /src/

#ENV COSMOSIS_SRC_DIR=

ENV MPLBACKEND Agg

COPY . /src/cosmopipe
RUN cd cosmopipe && \
    python -m pip install .

# ENV PYTHONPATH /src/cosmopipe:${PYTHONPATH}

RUN git clone http://bitbucket.org/joezuntz/cosmosis \
  && cd cosmosis \
  && git checkout \
  && git clone http://bitbucket.org/joezuntz/cosmosis-standard-library

#RUN cd cosmosis \
#  && source config/setup-cosmosis \
#  && make

WORKDIR /homedir/

RUN /sbin/ldconfig
