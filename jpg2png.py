import cv2
import os
from tqdm import tqdm

base = "datasets/custom/images"
for img in tqdm(os.listdir(base)):
    src = os.path.join(base, img)
    dst = os.path.join(base, img[:-4]+".png")
    cv2.imwrite(dst, cv2.imread(src))