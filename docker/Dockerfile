FROM ubuntu:20.04

RUN apt -y update && apt install -y apt-utils && echo yes

RUN DEBIAN_FRONTEND=noninteractive \
    apt install -y --no-install-recommends \
    make \
    gcc \
    git \
    wget \
    mpich \
    libmpich-dev \
    libbz2-dev \
    libcfitsio-dev \
    libcfitsio-bin \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN wget -nv https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    bash miniconda.sh -b -p /opt/conda && \
    conda config --add channels intel && \
    conda config --set always_yes yes && \
    conda update conda && \
    conda install python=3.7 pip numpy scipy && \
    conda clean -a -y

RUN conda install numpy mpi4py pyyaml cython gsl sphinx-rtd-theme

RUN for x in \
  git+https://github.com/adematti/pypescript \
  git+https://github.com/adematti/cosmopipe \
  ; do pip install $x; done \
  && rm -Rf /root/.cache/pip

WORKDIR /src/


#RUN git clone http://bitbucket.org/joezuntz/cosmosis \
#  && cd cosmosis \
#  && git checkout \
#  && git clone http://bitbucket.org/joezuntz/cosmosis-standard-library \
#  && make


ENV COSMOSIS_SRC_DIR=

ENV MPLBACKEND Agg

COPY . /src/cosmopipe

ENV PYTHONPATH /src/cosmopipe:${PYTHONPATH}

WORKDIR /homedir/
