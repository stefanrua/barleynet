#!/bin/bash
git submodule init
git submodule update
python3 -m venv venv
source venv/bin/activate
python3 -m pip install torch torchvision
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
mkdir models images tiles
