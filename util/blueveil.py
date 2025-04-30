import numpy as np
def measure_blue_veil(image):
    
    height, width, _ = image.shape
    count = 0

    for y in range(height):
        for x in range(width):
            b, g, r = image[y, x]

            if b > 60 and (r - 46 < g) and (g < r + 15):
                count += 1

    total_pixels = height * width
    normalized_score = count / total_pixels

    return normalized_score