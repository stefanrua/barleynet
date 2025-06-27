#!/bin/bash

config=unbiased-teacher-v2/configs/Faster-RCNN/alien-barley/old/all_samples.yaml
model=models/weights_faster-rcnn_model_0019999.pth
cut=true

for i in "$@"; do
    case $i in
        --fcos)
            config=unbiased-teacher-v2/configs/FCOS/alien-barley/all_samples.yaml
            model=models/fcos/model_0010399.pth
            shift
            ;;
        --nocut)
            cut=false
            shift
            ;;
    esac
done

source ~/miniforge3/etc/profile.d/conda.sh
conda activate ut2

if [ $cut = true ]; then
    python3 cut.py
fi
python3 unbiased-teacher-v2/train_net.py \
      --eval-only \
      --num-gpus 1 \
      --config $config \
      MODEL.WEIGHTS $model \
      DATASETS.TEST "('inference',)"
python3 reconstruct.py
