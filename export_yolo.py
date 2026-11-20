import argparse
import json
import os
import shutil
import cv2
import numpy as np
import yaml
from ultralytics.data.converter import merge_multi_segment

parser = argparse.ArgumentParser()
parser.add_argument("out")
parser.add_argument("--train", nargs="+", required=True)
parser.add_argument("--val", nargs="+", required=True)
parser.add_argument("--test", nargs="+", default=[])
args = parser.parse_args()
cfg = yaml.safe_load(open("config.yaml"))

for split in ["train", "val", "test"]:
    os.makedirs(os.path.join(args.out, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(args.out, f"labels/{split}"), exist_ok=True)
    split_meta = {}

    for src in getattr(args, split):
        coco = json.load(open(os.path.join(src, "coco_annotations.json")))
        meta = {m["file"]: m for m in json.load(open(os.path.join(src, "meta.json")))}
        anns = {}
        for ann in coco["annotations"]:
            anns.setdefault(ann["image_id"], []).append(ann)

        for img in coco["images"]:
            name = f"{os.path.basename(os.path.normpath(src))}_{os.path.basename(img['file_name'])}"
            w, h = img["width"], img["height"]
            yolo_lines = []
            for ann in anns.get(img["id"], []):
                if ann["area"] < cfg["min_visible_pixels"]:
                    continue
                # BlenderProc stores masks as uncompressed column-major RLE
                flat = np.zeros(h * w, np.uint8)
                pos = 0
                for k, count in enumerate(ann["segmentation"]["counts"]):
                    flat[pos:pos + count] = k % 2
                    pos += count
                contours, _ = cv2.findContours(flat.reshape(w, h).T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                segments = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
                polygon = np.concatenate(merge_multi_segment(segments)) if len(segments) > 1 else segments[0]
                yolo_lines.append("0 " + " ".join(f"{v:.5f}" for v in (polygon / [w, h]).ravel()) + "\n")

            shutil.copy(os.path.join(src, img["file_name"]), os.path.join(args.out, f"images/{split}", name))
            with open(os.path.join(args.out, f"labels/{split}", os.path.splitext(name)[0] + ".txt"), "w") as f:
                f.writelines(yolo_lines)
            split_meta[name] = meta[img["file_name"]]

    json.dump(split_meta, open(os.path.join(args.out, f"meta_{split}.json"), "w"), indent=1)
    print(f"{split}: {len(split_meta)} images")

dataset = {"path": os.path.abspath(args.out), "train": "images/train", "val": "images/val", "test": "images/test", "names": {0: cfg["class_name"]}}
yaml.safe_dump(dataset, open(os.path.join(args.out, "dataset.yaml"), "w"))
