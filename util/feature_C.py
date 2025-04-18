from skimage.segmentation import slic
from statistics import variance
import numpy as np

def slic_segmentation(mask, image, n_segments=30, compactness=0.1):

    # Run SLIC on the masked image
    slic_segments = slic(image,
                         n_segments=n_segments,
                         compactness=compactness,
                         sigma=1,
                         mask=mask,
                         start_label=1,
                         channel_axis=2)

    return slic_segments

def get_rgb_means(image, slic_segments):
    
    
    max_segment_id = np.max(slic_segments)
    rgb_means = []

    for i in range(1, max_segment_id + 1):

        segment_mask = (slic_segments == i)
        if np.any(segment_mask):  # Only process if segment exists
            rgb_mean = np.mean(image[segment_mask], axis=0)
            rgb_means.append(rgb_mean)
    
    return rgb_means if rgb_means else [np.array([0, 0, 0])]  # Handle empty case

def rgb_var(image, slic_segments):

    #input normalization if needed (ensures pizels values are 0-1)
    if image.dtype == np.uint8:
        image = image / 255.0  # Normalize to [0, 1]


    rgb_means = get_rgb_means(image, slic_segments)
    n = len(rgb_means) 

    if len(np.unique(slic_segments)) <= 2 or len(rgb_means) < 2:
        return 0, 0, 0

    red = []
    green = []
    blue = []
    for rgb_mean in rgb_means:
        red.append(rgb_mean[0])
        green.append(rgb_mean[1])
        blue.append(rgb_mean[2])

    red_var = variance(red, sum(red)/n)
    green_var = variance(green, sum(green)/n)
    blue_var = variance(blue, sum(blue)/n)

    max_possible_variance = 0.25  # for values in [0, 1] max variance is 0.25

    # Normalize to [0, 1]
    red_var_norm = red_var / max_possible_variance
    green_var_norm = green_var / max_possible_variance
    blue_var_norm = blue_var / max_possible_variance

    return red_var_norm, green_var_norm, blue_var_norm

