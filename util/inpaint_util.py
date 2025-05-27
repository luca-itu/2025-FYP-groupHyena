import cv2
import pandas as pd

def removeHair(img_org, img_gray, kernel_size=25, threshold=10, radius=3):
    # kernel for the morphological filtering
    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (kernel_size, kernel_size))

    # perform the blackHat filtering on the grayscale image to find the hair countours
    blackhat = cv2.morphologyEx(img_gray, cv2.MORPH_BLACKHAT, kernel)

    # intensify the hair countours in preparation for the inpainting algorithm
    _, thresh = cv2.threshold(blackhat, threshold, 255, cv2.THRESH_BINARY)

    # inpaint the original image depending on the mask
    img_out = cv2.inpaint(img_org, thresh, radius, cv2.INPAINT_TELEA)

    return img_out

df = pd.read_csv(r"C:\Users\karat\OneDrive\Dokumentumok\ITU\2_semester\Projects\2025-FYP-groupHyena\result\log_reg_features_results")
new_df = df[["img_id","ground_truth", "prediction", "probability_0", "probability_1"]]
new_df.to_csv("result_baseline.csv", index=False)
