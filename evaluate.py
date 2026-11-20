import argparse
import json
import os
import numpy as np
import yaml
from ultralytics import YOLO

parser = argparse.ArgumentParser()
parser.add_argument("weights")
parser.add_argument("dataset")
parser.add_argument("--split", default="test")
parser.add_argument("--conf", type=float, default=0.25)
parser.add_argument("--failures", help="write per-factor miss rates here (input for generate.py --focus)")
args = parser.parse_args()

cfg = yaml.safe_load(open("config.yaml"))
model = YOLO(args.weights)
metrics = model.val(data=os.path.join(args.dataset, "dataset.yaml"), split=args.split, plots=False, verbose=False)
print(f"{args.split}: box mAP50={metrics.box.map50:.3f} mAP50-95={metrics.box.map:.3f} mask mAP50-95={metrics.seg.map:.3f}")

# per-image matching: an object counts as detected if any prediction overlaps it with IoU >= 0.5
meta = json.load(open(os.path.join(args.dataset, f"meta_{args.split}.json")))
records, false_alarms, negatives = [], 0, 0
for name, m in meta.items():
    pred = model.predict(os.path.join(args.dataset, f"images/{args.split}", name), conf=args.conf, verbose=False)[0].boxes.xyxyn.cpu().numpy()
    rows = [np.array(line.split()[1:], float).reshape(-1, 2) for line in open(os.path.join(args.dataset, f"labels/{args.split}", os.path.splitext(name)[0] + ".txt"))]
    if not rows:
        negatives += 1
        false_alarms += len(pred) > 0
    for p in rows:
        gt = [*p.min(0), *p.max(0)]
        inter = np.clip(np.minimum(pred[:, 2], gt[2]) - np.maximum(pred[:, 0], gt[0]), 0, None) * np.clip(np.minimum(pred[:, 3], gt[3]) - np.maximum(pred[:, 1], gt[1]), 0, None)
        union = (pred[:, 2] - pred[:, 0]) * (pred[:, 3] - pred[:, 1]) + (gt[2] - gt[0]) * (gt[3] - gt[1]) - inter
        records.append((m, bool((inter / union >= 0.5).any())))

hits = np.array([hit for _, hit in records])
print(f"recall@conf{args.conf}: {hits.mean():.3f} ({hits.sum()}/{len(hits)}), false alarms on images without object: {false_alarms}/{negatives}")

failures = {}
for k, edges in cfg["factors"].items():
    values = np.array([m[k] for m, _ in records])
    bins = np.clip(np.digitize(values, edges) - 1, 0, len(edges) - 2)
    miss = [float(1 - hits[bins == b].mean()) if (bins == b).any() else 0.0 for b in range(len(edges) - 1)]
    failures[k] = {"edges": edges, "miss_rate": miss, "count": [int((bins == b).sum()) for b in range(len(edges) - 1)]}
    print(f"  {k:<11}" + "  ".join(f"[{edges[b]:g},{edges[b + 1]:g}): {miss[b]:.2f}" for b in range(len(edges) - 1)))

if args.failures:
    json.dump(failures, open(args.failures, "w"), indent=1)
