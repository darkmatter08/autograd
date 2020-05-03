from typing import List

import numpy as np


class Tensor:
    '''
    Holds an arbitrary-dimensional tensor, along with pointers to its dependencies.
    '''
    
    def __init__(self, data: np.ndarray, dependencies: List['Tensor']=None, requires_grad=False):
        self.data = data
        self.dependencies = dependencies
        self.shape = data.shape

        self._gradient = None
        self.requires_grad = requires_grad

    def __matmul__(self, right: 'Tensor') -> 'Tensor':
        """ @ operator.
        t3 = self @ right"""
        assert type(right) is Tensor
        result = self.data @ right.data
        return Tensor(result, dependencies=[self, right], requires_grad=(self.requires_grad or right.requires_grad))
     


    

        
    

        
