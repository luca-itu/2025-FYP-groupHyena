import numpy as np

def regionGrowing(image, seed, threshold):
    # Check if the image is RGB or grayscale 
    if len(image.shape) == 2:  # Grayscale image
        h, w = image.shape
        is_rgb = False
    elif len(image.shape) == 3:  # RGB image
        h, w, _ = image.shape  
        is_rgb = True
    else:
        raise ValueError("Unsupported image format")

    # Initialize a binary mask
    binary_mask = np.zeros((h, w), dtype=bool)
    stack = [seed]

    if is_rgb:
        region_mean = np.mean(image[seed], axis=-1)  
    else:
        region_mean = float(image[seed])  
    
    binary_mask[seed] = True
    count = 1
    
    # 4-connected neighbors: up, down, left, right
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    
    while stack:
        x, y = stack.pop()
        for dx, dy in neighbors:
            nx, ny = x + dx, y + dy
            if 0 <= nx < h and 0 <= ny < w and not binary_mask[nx, ny]:
                if is_rgb:
                    pixel_val = np.mean(image[nx, ny])  # Mean of R,G,B
                else:
                    pixel_val = float(image[nx, ny])
                
                if abs(pixel_val - region_mean) <= threshold:
                    binary_mask[nx, ny] = True
                    stack.append((nx, ny))
                    
                    # Update region mean dynamically
                    region_mean = (region_mean * count + pixel_val) / (count + 1)
                    count += 1

    # If input is RGB, return an RGB mask (foreground = original colors, background = black)
    if is_rgb:
        rgb_mask = np.zeros_like(image)  # Initialize as black image
        rgb_mask[binary_mask] = image[binary_mask]  # Copy original RGB pixels where mask is True
        return rgb_mask
    else:
        return binary_mask  # For grayscale, return binary mask
