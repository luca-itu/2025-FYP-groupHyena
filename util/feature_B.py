import numpy as np
from skimage import morphology
from skimage.measure import perimeter
from math import pi

def compactness(mask):
    A = np.sum(mask)
    l = perimeter(mask)  # Accurate perimeter

     # Handle edge cases
    if l == 0:
        if A == 0:
            return 0.0
        else:
            return 0.0

    compactness = (4 * pi * A) / (l ** 2)

    # Normalize the compactness to [0, 1] range
    max_compactness = 1  # This assumes a perfect circle, compactness max is 1.
    
    # Normalize by scaling the compactness relative to max value
    normalized_compactness = min(compactness / max_compactness, 1.0) 
    
    return normalized_compactness

#if low its irregular, if high its compact