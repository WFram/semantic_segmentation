FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

RUN mkdir -p /app/vins

COPY env.yml /app/vins

WORKDIR /app/vins

RUN apt-get update && apt-get install -y apt-transport-https
RUN apt-get update && apt-get install build-essential software-properties-common -y \
    sudo \
    cmake \
    libboost-all-dev \
    graphviz \
    libgl1-mesa-dev \
    libglew-dev \
    libhdf5-dev \
    libfreetype6-dev \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libavutil-dev \
    sed \
    gcc-9 \
    g++-9 \
    zlib1g \
    ninja-build \
    wget \
    ffmpeg \
    libgflags-dev \
    python3 \
    python3-pip \
    git \
    unzip \
    python3-dev \
    libopenblas-dev \
    bzip2 \
    ca-certificates

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh && \
    /bin/bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh

ENV PATH=$PATH:/opt/conda/bin

RUN conda env create -f env.yml && \
    conda init bash && \
    conda activate env && \
    echo "conda activate env" >> ~/.bashrc

RUN cd /app/vins && \
    git clone https://github.com/open-mmlab/mmcv.git && \
    cd mmcv && \
    pip install -r requirements/optional.txt && \
    pip install -v -e . && \
    cd ..
    # TODO: rm?

RUN cd /app/vins && \
    git clone -b main https://github.com/open-mmlab/mmsegmentation.git && \
    cd mmsegmentation && \
    pip install -v -e . && \
    cd ..
    # TODO: rm?