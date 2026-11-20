import json
import os
import shutil
import random

# paths
coco_json = "/path/to/blenderproc_output/coco_annotations.json" # for example "/Users/user/Desktop/Custom_Object_Detection/blenderproc_output/coco_annotations.json"
images_dir = "/path/to/blenderproc_output/images/" # for example "/Users/user/Desktop/Custom_Object_Detection/blenderproc_output/images/"
output_dir = "/path/to/created/yolo_dataset/" # for example "/Users/user/Desktop/Custom_Object_Detection/yolo_dataset/"

# split ratio
train_ratio = 0.7
val_ratio = 0.2
test_ratio = 0.1

# output directories
for split in ['train', 'val', 'test']:
    os.makedirs(os.path.join(output_dir, f"images/{split}"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, f"labels/{split}"), exist_ok=True)

with open(coco_json, 'r') as f:
    coco = json.load(f)

images = coco['images'].copy()
random.shuffle(images)

total = len(images)
train_end = int(total * train_ratio)
val_end = train_end + int(total * val_ratio)

splits = {}
splits['train'] = images[:train_end]
splits['val'] = images[train_end:val_end]
splits['test'] = images[val_end:]

print(f"Split: Train={len(splits['train'])}, Val={len(splits['val'])}, Test={len(splits['test'])}")

for split_name, split_images in splits.items():
    for img_data in split_images:
        img_id = img_data['id']
        img_filename = os.path.basename(img_data['file_name'])
        img_width = img_data['width']
        img_height = img_data['height']

        anns = [ann for ann in coco['annotations'] if ann['image_id'] == img_id]
        
        # COCO bbox to YOLO format
        yolo_lines = []
        for ann in anns:
            category_id = ann['category_id'] - 1
            bbox = ann['bbox']
            
            # YOLO format
            x_center = (bbox[0] + bbox[2] / 2) / img_width
            y_center = (bbox[1] + bbox[3] / 2) / img_height
            width = bbox[2] / img_width
            height = bbox[3] / img_height
            
            yolo_lines.append(f"{category_id} {x_center} {y_center} {width} {height}\n")
        
        src_img = os.path.join(images_dir, img_filename)
        dst_img = os.path.join(output_dir, f"images/{split_name}", img_filename)
        shutil.copy(src_img, dst_img)
        
        label_filename = os.path.splitext(img_filename)[0] + ".txt"
        label_path = os.path.join(output_dir, f"labels/{split_name}", label_filename)
        with open(label_path, 'w') as f:
            f.writelines(yolo_lines)

print(f"Conversion complete! Check {output_dir}")