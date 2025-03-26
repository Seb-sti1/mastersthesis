FROM ubuntu:22.04

LABEL maintainer="Seb-sti1"
LABEL version="2.0"
LABEL build="docker build -t sebsti1/mt_python -f docker/python.Dockerfile ."

SHELL ["/bin/bash", "-c"]

WORKDIR /app


RUN apt update && \
    # Install common deps
    apt -y install --no-install-recommends git build-essential libgl1 libglib2.0-0 wget nano ca-certificates zip unzip && \
    rm -rf /var/lib/apt/lists/* &&\
    update-ca-certificates &&\
    # install conda
    mkdir -p ~/miniconda3 &&\
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh &&\
    bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3 &&\
    rm ~/miniconda3/miniconda.sh &&\
    . ~/miniconda3/bin/activate &&\
    conda init bash

# create env for my scripts
RUN mkdir -p /app/mastersthesis && cd /app/mastersthesis &&\
    git clone https://github.com/Seb-sti1/mastersthesis.git /app/mastersthesis &&\
    . ~/miniconda3/bin/activate &&\
    conda create -n mastersthesis pip python=3.10 &&\
    conda activate mastersthesis &&\
    # install tools to build packages
    conda install -c nvidia cuda &&\
    pip install -r scripts/requirements.txt

# create env for omniglue
RUN mkdir -p /app/omniglue && cd /app/omniglue &&\
    git clone https://github.com/google-research/omniglue.git /app/omniglue &&\
    . ~/miniconda3/bin/activate &&\
    conda create -n omniglue pip python=3.10 &&\
    conda activate omniglue &&\
    pip install -e . &&\
    # fix cuDNN version (based on https://github.com/tensorflow/tensorflow/issues/60913#issuecomment-1707955579)
    pip install nvidia-cudnn-cu12==9.3.0.75 &&\
    # download models
    mkdir models && cd models &&\
    git clone https://github.com/rpautrat/SuperPoint.git &&\
    mv SuperPoint/pretrained_models/sp_v6.tgz . && rm -rf SuperPoint &&\
    tar zxvf sp_v6.tgz && rm sp_v6.tgz &&\
    wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_pretrain.pth &&\
    wget https://storage.googleapis.com/omniglue/og_export.zip &&\
    unzip og_export.zip && rm og_export.zip &&\
    # install benchmark deps
    pip install pandas scikit-image psutil tqdm opencv-contrib-python &&\
    ln -s /app/mastersthesis/scripts/matching/benchmark_omniglue.py

# create env for xfeat
RUN mkdir -p /app/xfeat && cd /app/xfeat &&\
    git clone https://github.com/verlab/accelerated_features.git /app/xfeat &&\
    . ~/miniconda3/bin/activate &&\
    conda create -n xfeat pip python=3.10 pytorch=2.0.0 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia &&\
    conda activate xfeat &&\
    conda install -c nvidia cuda &&\
    pip install kornia &&\
    # install benchmark deps
    pip install pandas scikit-image psutil tqdm opencv-contrib-python numpy==1.26.4 &&\
    ln -s /app/mastersthesis/scripts/matching/benchmark_xfeat.py

ENTRYPOINT ["/bin/bash"]