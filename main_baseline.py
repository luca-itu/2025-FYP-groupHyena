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
from util.classifier_LogReg import logistic_regression_classifier
from util.img_util import getImages

from joblib import Parallel, delayed


def extract_feature_baseline(img, img_id):
    
    #segmentation using region growing, finds the darkest pixel in the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    min_index = np.unravel_index(np.argmin(gray_img), gray_img.shape)

    segmentation_mask = regionGrowing(gray_img, min_index, threshold=20) #threshold set to lower, maybe do edge dialation

    #Feature extraction
    A_score = round(mean_asymmetry(segmentation_mask),3)
    B_score = round(compactness(segmentation_mask),3)
    red_var, green_var, blue_var = rgb_var(img, slic_segmentation(segmentation_mask, img))

    return {
        'img_id': img_id,
        'asymmetry_score': A_score,
        'border_score': B_score,
        'color_score_red': round(red_var, 3),
        'color_score_green': round(green_var, 3),
        'color_score_blue': round(blue_var, 3)
    }

def extract_df_baseline(folder_dir, n_jobs=1):
    images, img_ids = getImages(folder_dir)
    
    # Process images in parallel if n_jobs > 1
    features = Parallel(n_jobs=n_jobs)(
        delayed(extract_feature_baseline)(img, img_id) for img, img_id in zip(images, img_ids)
    )
    
    return pd.DataFrame(features)

#df = extract_df_baseline(r"data")
#labels_df = pd.read_csv(r"dataset.csv")  

#df_with_labels = df.merge(labels_df, on="img_id", how="inner")
#result = logistic_regression_classifier(df_with_labels, use_smote=False) # for testing purposes

#print(result[2])
