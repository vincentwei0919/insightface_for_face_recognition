#!/bin/bash
python verification.py --data-dir ../dataset/images --model ../models/model-r50-am-lfw/model,0000 --gpu 0 --target valid --nfolds 10;

