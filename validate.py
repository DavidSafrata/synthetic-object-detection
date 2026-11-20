import argparse
import glob
import os
import random
import cv2
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("dataset")
parser.add_argument("--preview", default=None)
args = parser.parse_args()

for split in ["train", "val", "test"]:
    images = sorted(glob.glob(os.path.join(args.dataset, f"images/{split}/*")))
    if not images:
        continue
    label_dir = os.path.join(args.dataset, f"labels/{split}")
    stems = {os.path.splitext(os.path.basename(p))[0] for p in images}
    issues = [f"label without image: {p}" for p in glob.glob(f"{label_dir}/*.txt") if os.path.splitext(os.path.basename(p))[0] not in stems]
    boxes, negatives = [], 0

    for path in images:
        label = os.path.join(label_dir, os.path.splitext(os.path.basename(path))[0] + ".txt")
        if not os.path.exists(label):
            issues.append(f"missing label: {label}")
            continue
        rows = [list(map(float, line.split())) for line in open(label) if line.strip()]
        negatives += not rows
        for r in rows:
            if r[0] != 0 or len(r) < 5 or len(r) % 2 == 0 or min(r[1:]) < 0 or max(r[1:]) > 1:
                issues.append(f"invalid row: {label}")
            elif len(r) == 5:
                boxes.append([r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2])
            else:
                p = np.array(r[1:]).reshape(-1, 2)
                boxes.append([*p.min(0), *p.max(0)])

    boxes = np.array(boxes)
    size = np.sqrt((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]))
    center = (boxes[:, :2] + boxes[:, 2:]) / 2
    print(f"{split}: {len(images)} images, {len(boxes)} objects, {negatives} without object, {len(issues)} issues")
    print(f"  box size (sqrt area / image) p5={np.percentile(size, 5):.3f} p50={np.median(size):.3f} p95={np.percentile(size, 95):.3f}")
    print(f"  centers within middle 20% of image: {np.mean(np.all(np.abs(center - 0.5) < 0.1, axis=1)):.0%}")
    print(f"  touching image border: {np.mean((boxes[:, :2] < 0.002).any(1) | (boxes[:, 2:] > 0.998).any(1)):.0%}")
    for issue in issues[:10]:
        print("  " + issue)

if args.preview:
    random.seed(0)
    tiles = []
    for path in random.sample(sorted(glob.glob(os.path.join(args.dataset, "images/train/*"))), 16):
        img = cv2.imread(path)
        h, w = img.shape[:2]
        label = path.replace(f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}").rsplit(".", 1)[0] + ".txt"
        for line in open(label):
            r = np.array(line.split()[1:], float)
            if len(r) == 4:
                r = np.array([r[0] - r[2] / 2, r[1] - r[3] / 2, r[0] + r[2] / 2, r[1] - r[3] / 2, r[0] + r[2] / 2, r[1] + r[3] / 2, r[0] - r[2] / 2, r[1] + r[3] / 2])
            p = (r.reshape(-1, 2) * [w, h]).astype(np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [p], (0, 0, 255))
            img = cv2.addWeighted(overlay, 0.4, img, 0.6, 0)
            cv2.rectangle(img, tuple(map(int, p.min(0))), tuple(map(int, p.max(0))), (0, 255, 0), 2)
        tiles.append(cv2.resize(img, (320, 320)))
    grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, 16, 4)])
    cv2.imwrite(args.preview, grid)
    print(f"preview written to {args.preview}")
