from skimage import io
import numpy as np
import cv2
from multiprocessing import Pool

def process_block(block, tile_list, i, j):
    """
    Process a block of the image to find the best matching tile and resize it.

    Parameters:
    block (ndarray): The block of the image to be processed.
    tile_list (list): List of tiles with their average colors.
    i (int): The row index of the block in the original image.
    j (int): The column index of the block in the original image.

    Returns:
    tuple: The row index, column index, and the resized tile image.
    """
    bh, bw = block.shape[:2]
    avg_color = np.mean(block, axis=(0,1))
    
    best_tile_img, _ = min(
                tile_list,
                key=lambda x: np.linalg.norm(avg_color - x[1])
            )
            
    tile_resized = cv2.resize(best_tile_img, (bw, bh))
            
    return i, j, tile_resized
    
def mosaiker(image_path, block_size, tile_list):
    """
    Create a mosaic image by dividing the input image into blocks and replacing each block with the best matching tile.

    Parameters:
    image_path (str): Path to the input image.
    block_size (int): Size of the blocks to divide the image into.
    tile_list (list): List of tiles with their average colors.

    Returns:
    ndarray: The mosaic image.
    """
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
    """
    Create a mosaic image using parallel processing to speed up the block processing.

    Parameters:
    image_path (str): Path to the input image.
    block_size (int): Size of the blocks to divide the image into.
    tile_list (list): List of tiles with their average colors.
    num_processes (int): Number of processes to use for parallel processing.

    Returns:
    ndarray: The mosaic image.
    """
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
        
    for (i, j, tile_resized) in results:
        bh, bw = tile_resized.shape[:2]
        new_image[i: i + bh, j: j + bw] = tile_resized
        
    return new_image
