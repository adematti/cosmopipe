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
    python3 \
    python3-dev \
    python3-pip \
    python3-pil \
    python3-tk \
    # # Remove APT files
    && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

RUN ln -s /usr/bin/python3 /usr/bin/python
RUN ln -s /usr/bin/pip3 /usr/bin/pip

RUN for x in \
  scipy \
  numpy \
  matplotlib \
  mpi4py \
  pyyaml \
  fitsio \
  tabulate \
  git+https://github.com/adematti/cosmoprimo \
  graphviz \
#  graphviz-dev \
  coverage \
  sphinx \
  sphinx-rtd-theme \
  ; do pip install $x; done \
  && rm -Rf /root/.cache/pip

WORKDIR /src/

RUN git clone https://github.com/adematti/pypescript \
  && cd pypescript \
  && python -m pip install . \
  && cd template_lib \
  && python -m pip install .

# RUN git clone https://github.com/adematti/pypescript \
#  && cd pypescript/template_lib \
#  && python -m pip install .

RUN for x in \
  git+https://github.com/CobayaSampler/cobaya \
  ; do pip install $x; done \
  && rm -Rf /root/.cache/pip

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
