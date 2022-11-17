#!/bin/bash

python3 cut.py
python3 unbiased-teacher-v2/train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config unbiased-teacher-v2/configs/Faster-RCNN/alien-barley/all_samples.yaml \
      MODEL.WEIGHTS models/model_0107999.pth \
      DATASETS.TEST "('inference',)"
python3 reconstruct.py
