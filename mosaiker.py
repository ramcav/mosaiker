from skimage import io
import numpy as np
import cv2
from multiprocessing import Pool

def process_block(block, tile_list, i, j):
    bh, bw = block.shape[:2]
    avg_color = np.mean(block, axis=(0,1))
    
    best_tile_img, _ = min(
                tile_list,
                key=lambda x: np.linalg.norm(avg_color - x[1])
            )
            
    tile_resized = cv2.resize(best_tile_img, (bw, bh))
            
    return i, j, tile_resized
    
def mosaiker(image_path, block_size, tile_list):
    image = io.imread(image_path)
    h, w, _ = image.shape
   
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            
            block = image[i : i + block_size, j : j + block_size]
            bh, bw = block.shape[:2]
            
            _, _, tile_resized = process_block(block, tile_list, i, j)
            
            new_image[i : i + bh, j : j + bw] = tile_resized

    return new_image


def parallel_mosaiker(image_path, block_size, tile_list, num_processes):
    image = io.imread(image_path)
    h, w, _ = image.shape
   
    new_image = np.zeros((h, w, 3), dtype=np.uint8)
    blocks = []
    
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            
            block = image[i : i + block_size, j : j + block_size]
            blocks.append((block, tile_list, i, j))
            
    with Pool(processes=num_processes) as pool:
        results = pool.starmap(process_block, blocks)
        
    for (i, j, tile_resized)in results:
        bh, bw = tile_resized.shape[:2]
        new_image[i: i + bh, j: j+bw] = tile_resized
        
    return new_image
