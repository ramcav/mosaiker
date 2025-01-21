from mosaiker import mosaiker, parallel_mosaiker
from tiles import load_and_preprocess
import cv2
import os
import random
import re

def main():
    path = 'https://t4.ftcdn.net/jpg/07/18/12/87/360_F_718128776_nJReWqPkf5qF4Y5na8ZqGWAbdCJTpczZ.jpg'
    
    tile_list = load_and_preprocess()
    
    block_size = 8
    cpu_count = os.cpu_count()
    
    mosaic_image = parallel_mosaiker(path, block_size, tile_list, cpu_count)
    
    match = re.search(r'https://(.*?)/', path)
    
    if match:
        domain_name = match.group(1).replace('.', '_')
    else:
        domain_name = 'unknown_domain'
    
    cv2.imwrite(f'results/mosaic_result-{block_size}-{domain_name}.jpg', cv2.cvtColor(mosaic_image, cv2.COLOR_RGB2BGR))
    
    
if __name__ == '__main__':
    main()