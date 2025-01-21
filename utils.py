import os

def is_folder_empty(path: str) -> bool:
    
    if len(os.listdir(path)) == 0: return True

    return False