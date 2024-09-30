FROM nvcr.io/nvidia/pytorch:24.03-py3

RUN apt-get update && apt-get install -y default-jdk zip

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN python3 -c "import urllib.request; \
    urllib.request.urlretrieve('https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh', 'miniconda.sh')" && \
    /bin/bash miniconda.sh -b -p $CONDA_DIR && \
    rm miniconda.sh && \
    $CONDA_DIR/bin/conda init && \
    $CONDA_DIR/bin/conda install python=3.9.18 -y
ENV PATH=$CONDA_DIR/bin:$PATH

COPY ./torchserve/environment.yml $CONDA_DIR/environment.yml
RUN conda update conda
RUN conda env update --name base -f $CONDA_DIR/environment.yml

RUN mkdir -p /opt/serve && \
    git clone https://github.com/pytorch/serve.git /opt/serve

WORKDIR /opt/serve

RUN python ./ts_scripts/install_dependencies.py --cuda=cu121 && \
    mkdir -p /opt/serve/model-store && mkdir -p /opt/serve/logs