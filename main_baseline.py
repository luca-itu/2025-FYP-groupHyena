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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score


def extract_feature_baseline(img, img_id):
    
    #segmentation using region growing, finds the darkest pixel in the image
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    min_index = np.unravel_index(np.argmin(gray_img), gray_img.shape)

    segmentation_mask = regionGrowing(gray_img, min_index, threshold=20)

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

if __name__ == "__main__":

    df = extract_df_baseline(r"data")
    labels_df = pd.read_csv(r"dataset.csv")[["img_id", "ground_truth"]] 

    # merge dataset and images
    df_with_labels = df.merge(labels_df, on="img_id", how="inner")
    result = logistic_regression_classifier(df_with_labels, use_smote=False)  

    # extract results
    ground_truth = result[0]
    predictions = result[1]
    accuracy = result[2]
    probability = result[3]
    prob_class0 = probability[:, 0]  
    prob_class1 = probability[:, 1]

    # save the results to a dataframe
    resulting_df = pd.DataFrame({
    'img_id': df_with_labels.loc[ground_truth.index, 'img_id'],
    'ground_truth': ground_truth,
    'prediction': predictions,
    'probability_0': np.round(prob_class0, 2),
    'probability_1': np.round(prob_class1, 2)
    })

    #save the results to a csv
    resulting_df.to_csv(r"result/results_logreg_ta.csv", index=False)

    # extract confusion matrix
    conf_mat = confusion_matrix(y_true=ground_truth, y_pred=predictions)

    # extract
    class_report = classification_report(ground_truth, predictions)

    # extract AUC score
    roc_auc = roc_auc_score(ground_truth, prob_class1)

    # print
    print(f"Test accuracy: {accuracy}")
    print(f"AUC = {roc_auc}")
    print("Confusion matrix:\n", conf_mat)
    print("Classification report:\n", class_report)

