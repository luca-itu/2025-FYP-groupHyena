import os
from os import listdir
import numpy as np
import cv2
import pandas as pd
from util.regionGrowing_util import regionGrowing

from util.feature_A import mean_asymmetry
from util.feature_B import compactness
from util.feature_C import rgb_var
from util.feature_C import slic_segmentation

from util.classifier_kNN import knn_algorithm_smote

def extract_features(folder_dir):
  feature_list = []
  for filename in os.listdir(folder_dir):
      if filename.endswith(".png"):
          lesion_id = filename.split('.')[0] 
          file_path = os.path.join(folder_dir, filename)
          img = cv2.imread(file_path)
          img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

          #resize if needed
          max_size = 256
          h, w = img.shape[:2]
          scale = min(max_size / w, max_size / h)
          if scale < 1:
              new_size = (int(w * scale), int(h * scale))
              img = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

          # segmentation using region growing, finds the darkest pixel in the image
          gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
          min_index = np.unravel_index(np.argmin(gray_img), gray_img.shape)
        
          segmentation_mask = regionGrowing(gray_img, min_index, threshold=20) #threshold set to lower, maybe do edge dialation
          
          #mask for RGB image for feature_C
          segmentation_mask_rgb = regionGrowing(img, min_index, threshold=20) 

          #Feature extraction
          A_score = round(mean_asymmetry(segmentation_mask),3)
          B_score = round(compactness(segmentation_mask),3)
          red_var, green_var, blue_var = rgb_var(img, slic_segmentation(segmentation_mask_rgb, img))

          feature_list.append({
                'lesion_id': lesion_id,  # Use lesion ID from the filename
                'asymmetry_score': A_score,
                'border_score': B_score,
                'color_score_red': round(red_var, 3),
                'color_score_green': round(green_var, 3),
                'color_score_blue': round(blue_var, 3)
            })
  df = pd.DataFrame(feature_list)
  return df

#loading the dataset (uses extract_features function from main_baseline.py)
df = extract_features(r"2025-FYP-groupHyena/data")

#ensure the lesion_id column is of type string
df['lesion_id'] = df['lesion_id'].astype(str)

#load the csv file containing the labels
labels_df = pd.read_csv(r"C:\Users\sarac\OneDrive\Desktop\PDS\2025-FYP-groupHyena\dataset.csv")  

# Merge only the matching data points
df_with_labels = df.merge(labels_df, on="lesion_id", how="inner")

#call the knn model
result_smote = knn_algorithm_smote(df_with_labels, k=5, distance_metric='euclidean', use_smote=True)
test_accuracy_smote = result_smote[0]  # Extract the accuracy score from the result tuple or whatever we need
