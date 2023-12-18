from pycocotools.coco import COCO
import cv2
import numpy as np
import os
from tqdm import tqdm

# Load COCO annotations
coco_annotation_file = '/share/liangyingping/RAFT/datasets/custom/annotations/instances_val2017.json'
coco = COCO(coco_annotation_file)

# Directory to save masks
mask_directory = 'datasets/custom/masks'
os.makedirs(mask_directory, exist_ok=True)

# Loop through each image in the dataset
for img_id in tqdm(coco.imgs):
    img_info = coco.imgs[img_id]
    img_width = img_info['width']
    img_height = img_info['height']

    # Create a blank mask image
    mask = np.zeros((img_height, img_width), dtype=np.uint8)

    # Draw each polygon annotation on the mask with a unique fill value
    instance_index = 1  # Start from 1 as 0 is typically considered background
    for ann in coco.imgToAnns[img_id]:
        if 'segmentation' in ann:
            for segmentation in ann['segmentation']:
                if len(segmentation) and not isinstance(segmentation, str):
                    polygon = np.array(segmentation).reshape((-1, 1, 2)).astype(np.int32)
                    cv2.fillPoly(mask, [polygon], color=instance_index)
                    instance_index += 1

    # Save or use the mask 000000037777
    mask_filename = os.path.join(mask_directory, '{:012d}.png'.format(img_id))
    cv2.imwrite(mask_filename, mask)
    mask_filename = os.path.join(mask_directory, '{:012d}_mask.png'.format(img_id))
    cv2.imwrite(mask_filename, mask * 50)
