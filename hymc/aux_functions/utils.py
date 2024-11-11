import os
import random
from typing import List

import numpy as np


def create_folder(folder_path: str):
    """Create a folder to store the results.

    Checks if the folder where one will store the results exist. If it does not, it creates it.

    Parameters
    ----------
    folder_path : str
        Path to the location of the folder
    
    """
    if not os.path.exists(folder_path):
        # Create the folder
        os.makedirs(folder_path)
        print(f"Folder '{folder_path}' created successfully.")
    else:
        print(f"Folder '{folder_path}' already exists.")

def read_basins_id(path_entities: str) -> List[str]:
    """Read the basins id from a text file.

    Parameters
    ----------
    path_entities : str
        Path to the file with the basins id

    Returns
    -------
    List[str]
        List with the basins id
    
    """
    entities_ids = np.loadtxt(path_entities, dtype="str").tolist()
    entities_ids = [entities_ids] if isinstance(entities_ids, str) else entities_ids

    return entities_ids

def set_random_seed(seed: int=None):
    """Set a seed for various packages to be able to reproduce the results.

    Parameters
    ----------
    seed : int
        Number of the seed
    
    """
    if seed is None:
        seed = int(np.random.uniform(low=0, high=1e6))

    random.seed(seed)
    np.random.seed(seed)


def write_report(file_path: str, text: str):
    """Write a given text into a text file.
    
    If the file where one wants to write does not exists, it creates a new one.

    Parameters
    ----------
    file_path : str
        Path to the file where 
    text : str
        Text that wants to be added
    
    """
    if os.path.exists(file_path):
        append_write = "a" # append if already exists   
    else:
        append_write = "w" # make a new file if not

    highscore = open(file_path , append_write)
    highscore.write(text + "\n")
    highscore.close()


