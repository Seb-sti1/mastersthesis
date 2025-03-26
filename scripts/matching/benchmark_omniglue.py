#!/usr/bin/env python3


"""


Start a docker container from the root folder of the omniglue repo (don't foget to change /home/seb-sti1/sebspace/mastersthesis to the correct path)

docker run -it --name omniglue --gpus all --privileged --device /dev/dri:/dev/dri -e DISPLAY=$DISPLAY -e XAUTHORITY=/root/.Xauthority -v /tmp/.X11-unix:/tmp/.X11-unix -v $HOME/.Xauthority:/root/.Xauthority -v /home/seb-sti1/sebspace/mastersthesis:/app -v $(pwd):/workdir ubuntu:22.04

Now in the docker

# install deps
apt update && apt install -y libgl1 libglib2.0-0 wget nano

# install miniconda (needed because the docker image probided by conda is broken as it uses debian 9)
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh
source ~/miniconda3/bin/activate
conda init bash  # you don't need to restart shit

# install package
conda create -n omniglue pip python=3.10
conda activate omniglue
cd /workdir && pip install -e .

conda install -c nvidia cuda-nvcc # not sure
conda install -c nvidia cudnn=9.3.0 # not sure

# fix cuDNN version (based on https://github.com/tensorflow/tensorflow/issues/60913#issuecomment-1707955579)
pip install nvidia-cudnn-cu12==9.3.0.75

# "Copy" this file in the right place 
ln -s /app/scripts/matching/benchmark_omniglue.py

# install benchmark deps
pip install pandas scikit-image psutil

# Run the benchmark (first you can use the `python demo.py res/demo1.jpg res/demo2.jpg` to test if omniglue and its dependencies are correctly installed)
python benchmark_omniglue.py


# old tips

# install cuda
apt update && apt install -y nvidia-cuda-toolkit

# fix path to cuda (inspired from https://github.com/tensorflow/tensorflow/issues/58681#issuecomment-1535032734)
ln -s /usr/lib/cuda /usr/local/cuda
"""

import sys
import time

import numpy as np

sys.path.append("/app/omniglue")
import omniglue
from omniglue import utils

package_folder = '/app/mastersthesis'
sys.path.append(package_folder)

from scripts.matching.test_framework import benchmark
from scripts.utils.datasets import resize_image

og = None
crop = 20


def compare(i, im1, im2):
    im1 = resize_image(im1, 400, 400)
    im2 = resize_image(im2, 400, 400)
    im1 = np.copy(im1)
    im2 = np.copy(im2)

    t_match = time.time_ns()
    mkpts_0, mkpts_1, scores = og.FindMatches(im1, im2)
    t_match = time.time_ns() - t_match

    return (mkpts_0, mkpts_1,
            (t_match,
             np.min(scores), np.max(scores), np.mean(scores), np.std(scores),
             mkpts_0.shape[0]))


def main() -> None:
    global og

    og = omniglue.OmniGlue(
        og_export="./models/og_export",
        sp_export="./models/sp_v6",
        dino_export="./models/dinov2_vitb14_pretrain.pth",
    )
    benchmark(compare,
              0.5,
              ["match duration",
               "min scores", "max scores", "mean scores", "std scores",
               "number of matched point"],
              "omniglue")


if __name__ == "__main__":
    main()
