import argparse
import os
import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("name")
args = parser.parse_args()

cfg = yaml.safe_load(open("config.yaml"))["train"]
model = YOLO(cfg.pop("model"))
model.train(data=os.path.join(args.dataset, "dataset.yaml"), project=os.path.abspath("runs"), name=args.name, exist_ok=True, seed=0, **cfg)
