import os
from os import listdir
import numpy as np
import cv2
from util.feature_A import mean_asymmetry
from util.regionGrowing_util import regionGrowing
import pandas as pd
from util.feature_C import *
from util.feature_B import compactness

base_dir = os.path.dirname(__file__)
folder_dir = os.path.join(base_dir, "data")

input_csv = r'2025-FYP-groupHyena\dataset_chosen.csv'
df = pd.read_csv(input_csv)
#A_score = []
#B_score = []
#C_score = []

for filename in os.listdir(folder_dir):
    if filename.endswith(".png"):
        file_path = os.path.join(folder_dir, filename)
        img = cv2.imread(file_path, 0)
        max_size = 256
        h, w = img.shape[:2]
        scale = min(max_size / w, max_size / h)
        if scale < 1:
            new_size = (int(w * scale), int(h * scale))
            img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # find the darkest pixel
    min_index = np.unravel_index(np.argmin(img), img.shape)

    # apply region growing segmentation using the darkest pixel as the seed
    segmentation_mask = regionGrowing(img, min_index, threshold=20) #threshold set to lower, maybe do edge dialation

  #  A_score.append(round(mean_asymmetry(segmentation_mask),3))
  #  B_score.append(round(compactness(segmentation_mask),3))
  #  red_var, green_var, blue_var = rgb_var(img, slic_segmentation(segmentation_mask, img))
  #  C_score.append(round(red_var,3), round(green_var,3), round(blue_var,3))

#df['asymmetry_score'] = A_score
#df['border_score'] = B_score
#df['color_score'] = C_score
#df.to_csv(input_csv, index = False)
#print('dataset updated successfully!')