#!/bin/bash
git submodule init
git submodule update
python3 -m venv venv
source venv/bin/activate
python3 -m pip install torch torchvision
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'
mkdir models images tiles
wget -P models/ https://a3s.fi/swift/v1/AUTH_4b8657d0f1c44e83bae2cd293af06d2a/barleynet-models/model_0107999.pth
