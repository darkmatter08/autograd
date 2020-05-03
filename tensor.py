from typing import List

import numpy as np


class Tensor:
    '''
    Holds an arbitrary-dimensional tensor, along with pointers to its dependencies.
    '''
    
    def __init__(self, data: np.ndarray, dependencies: List = None):
        self.data = data
        self.dependencies = dependencies
