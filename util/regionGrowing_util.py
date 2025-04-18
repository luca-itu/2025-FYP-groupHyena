import numpy as np

def regionGrowing(image, seed, threshold):

    # Check if the image is RGB or grayscale 
    if len(image.shape) == 2:  # Grayscale image
        h, w = image.shape
    elif len(image.shape) == 3:  # RGB image
        h, w, _ = image.shape  
    else:
        raise ValueError("Unsupported image format")

    mask = np.zeros((h, w), dtype=bool)
    region_pixels = []
    stack = [seed]

    if len(image.shape) == 2:  # Grayscale image
        region_mean = float(image[seed])  # Grayscale pixel is a scalar
    elif len(image.shape) == 3:  # RGB image
        # Compute the mean of the RGB channels for the seed pixel
        region_mean = float(np.mean(image[seed]))  # Average over R, G, B channels
    else:
        raise ValueError("Unsupported image format")
    
    # initialize the region's intensity mean with the seed value
    count = 1
    
    mask[seed] = True
    region_pixels.append(seed)
    
    # 4-connected neighbors: up, down, left, right
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        x, y = stack.pop()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and mask[nx, ny] == 0:
                if len(image.shape) == 2:  # Grayscale image
                    pixel_val = float(image[nx, ny])
                elif len(image.shape) == 3:  # RGB image
                    # Compute the mean of the RGB channels for the current pixel
                    pixel_val = float(np.mean(image[nx, ny]))  # Average over R, G, B channels
                else:
                    raise ValueError("Unsupported image format")
                
                # compare the pixel intensity to the current region mean
                if abs(pixel_val - region_mean) <= threshold:
                    mask[nx, ny] = 1
                    stack.append((nx, ny))
                    region_pixels.append((nx, ny))
                    
                    # update region mean dynamically
                    region_mean = (region_mean * count + pixel_val) / (count + 1)
                    count += 1
                    
    return mask
