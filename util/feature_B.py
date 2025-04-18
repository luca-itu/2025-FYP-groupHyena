import numpy as np
from skimage import morphology
from skimage.measure import perimeter
from math import pi

def compactness(mask):
    A = np.sum(mask)
    l = perimeter(mask)  # Accurate perimeter

    compactness = (4 * pi * A) / (l ** 2)

    return compactness

#if low its irregular, if high its compact