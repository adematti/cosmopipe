FROM joezuntz/cosmosis:2.1
# comes from https://bitbucket.org/joezuntz/cosmosis-docker/src/master/base-image/Dockerfile

WORKDIR /

RUN git clone http://bitbucket.org/joezuntz/cosmosis \
  && cd cosmosis \
  && git checkout \
  && git clone http://bitbucket.org/joezuntz/cosmosis-standard-library \
  && make

RUN apt-get update
# tkinter otherwise issue in matplotlib
RUN DEBIAN_FRONTEND=noninteractive \
  apt install -y --no-install-recommends \
  python3-tk \
  graphviz \
  graphviz-dev \
  # # Remove APT files
  && apt-get clean && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*

# already in cosmosis but needs update
RUN for x in \
  git+https://github.com/CobayaSampler/cobaya \
  pygraphviz \
  psutil \
  coverage \
  sphinx \
  sphinx-rtd-theme \
#  git+https://github.com/adematti/pypescript \
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

ENV COSMOSIS_SRC_DIR=

ENV MPLBACKEND Agg

COPY . /src/cosmopipe

ENV PYTHONPATH /src/cosmopipe:${PYTHONPATH}

WORKDIR /homedir/

#git clone https://github.com/lesgourg/class_public.git class
#cd class
#make
