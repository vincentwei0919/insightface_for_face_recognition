#!/usr/bin/env bash
python -u  train_softmax_my.py --prefix ../my_model/ --pretrained ../models/model-r50-am-lfw/model,0000 --loss-type 4 \
--margin-m 0.1 --data-dir ../dataset/images --per-batch-size 32 --version-se 1 --verbose 1000 \
--target valid --margin-s 64 --emb-size 512

