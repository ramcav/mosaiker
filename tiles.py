import torchvision
import torchvision.transforms as transforms
import pickle
import numpy as np
from utils import is_folder_empty
import os

def download_tiles() -> None:
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    torchvision.datasets.CIFAR10(
        root='./tiles', train=True, 
        download=True, transform=transform
    )
    
def load_batch(path: str) -> dict:
    with open(path, 'rb') as f:
        batch_dict = pickle.load(f, encoding='bytes')
        
    return batch_dict[b'data']

def reshape_cifar_data(data_array: np.ndarray) -> np.ndarray:
    
    reshaped = data_array.reshape(-1, 3, 32, 32)
    transposed = reshaped.transpose(0, 2, 3, 1) # This swaps the 32 to be in the second dimension which is needed for the proper formatting of the image
    
    return transposed

def load_and_preprocess():
    if is_folder_empty('tiles'):
        download_tiles()
        
    base_path = 'tiles/cifar-10-batches-py'

    files = os.listdir(base_path)
    
    files.remove('readme.html')
    files.remove('batches.meta')
    
    tile_list = []
    
    for filepath in files:
        data = load_batch(f'{base_path}/{filepath}')
        images = reshape_cifar_data(data)
        
        for image in images:
            avg_color = np.mean(image, axis=(0, 1))
            tile_list.append((image,avg_color))
            
    return tile_list
    
    