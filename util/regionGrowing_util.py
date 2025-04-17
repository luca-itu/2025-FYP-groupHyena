import numpy as np

def regionGrowing(image, seed, threshold):
    h, w = image.shape
    mask = np.zeros((h, w), dtype=bool)
    region_pixels = []
    stack = [seed]
    
    # initialize the region's intensity mean with the seed value
    region_mean = float(image[seed])
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
                pixel_val = int(image[nx, ny])
                
                # compare the pixel intensity to the current region mean
                if abs(pixel_val - region_mean) <= threshold:
                    mask[nx, ny] = 1
                    stack.append((nx, ny))
                    region_pixels.append((nx, ny))
                    
                    # update region mean dynamically
                    region_mean = (region_mean * count + pixel_val) / (count + 1)
                    count += 1
                    
    return mask
