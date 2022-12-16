#!/bin/bash

if [ ! -d venv ]; then
    ./setup.sh
fi

config=unbiased-teacher-v2/configs/Faster-RCNN/alien-barley/all_samples.yaml
model=models/faster-rcnn/model_0061999.pth
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

source venv/bin/activate
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
