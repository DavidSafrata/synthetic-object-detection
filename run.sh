#!/bin/sh
set -e

python download_hdris.py
blenderproc run generate.py train 1000 data/train --seed 0
blenderproc run generate.py val 300 data/val --seed 1
blenderproc run generate.py test 400 data/test --seed 2
python export_yolo.py datasets/base --train data/train --val data/val --test data/test
python validate.py datasets/base --preview datasets/base/preview.jpg
python train.py datasets/base base
python evaluate.py runs/base/weights/last.pt datasets/base --split val --failures failures.json

# failure-driven round, compared against the same number of uniformly sampled images
blenderproc run generate.py train 400 data/train_focus --focus failures.json --seed 3
blenderproc run generate.py train 400 data/train_uniform --seed 4
python export_yolo.py datasets/focus --train data/train data/train_focus --val data/val --test data/test
python export_yolo.py datasets/uniform --train data/train data/train_uniform --val data/val --test data/test
python train.py datasets/focus focus
python train.py datasets/uniform uniform
for run in base uniform focus; do python evaluate.py runs/$run/weights/last.pt datasets/$run --split test; done
