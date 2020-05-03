from typing import List

import numpy as np


class Tensor:
    '''
    Holds an arbitrary-dimensional tensor, along with pointers to its dependencies.
    '''
    
    def __init__(self, data: np.ndarray, dependencies: List['Tensor']=None, requires_grad=False):
        self.data = data
        self.dependencies = dependencies
        if dependencies is None:
            self.dependencies = []
        self.shape = data.shape

        # self._gradient follows numerator convention
        self._gradient = np.zeros_like(self.data.T)
        
        self.requires_grad = requires_grad


    def backward(self, inbound_grad):
        for dependency in self.dependencies:
            grad_update = dependency.grad_fun(inbound_grad)
            assert (grad_update.shape == dependency.tensor._gradient.shape)
            dependency.tensor._gradient += grad_update
            dependency.tensor.backward(dependency.tensor._gradient)
            
        
    def __matmul__(self, right: 'Tensor') -> 'Tensor':
        """ @ operator.
        t3 = self @ right"""

        return mat_mul(self, right) #.forward(self, right)


class Dependency:
    def __init__(self, tensor, grad_fun):
        self.tensor = tensor
        self.grad_fun = grad_fun

    

# TODO: Encapsulate in a MatrixMultiply class, to make computation graph explicit
def mat_mul(left: 'Tensor', right: 'Tensor'):
    assert type(right) is Tensor
    assert type(left) is Tensor
    
    result = left.data @ right.data
    
    def left_grad(inbound_grad: np.ndarray):
        dL_d_left = right.data @ inbound_grad
        # assert (left._gradient.shape == dL_d_left.shape)
        return dL_d_left

    def right_grad(inbound_grad: np.ndarray):
        dL_d_right = inbound_grad @ left.data
        # assert (right._gradient.shape == dL_d_right.shape)
        return dL_d_right
    
    dependencies = [Dependency(left, left_grad),
                    Dependency(right, right_grad)]
    
    return Tensor(result,
                  dependencies,
                  requires_grad=(left.requires_grad or right.requires_grad))



'''
def backward(inbound_grad: np.ndarray):

        """
        left: (l1, l2)
        right: (l2, 1)

        self = left @ right
           (l1, 1)
        left is a matrix
        right is a col vector

        L is everything "downstream" of self. 
        dL/d(left) = right @ dL/d(self)
         (l2, l1)  = (l2, 1) @ (1, l1)

        dL/d(right) = dL/d(self) @ left
         (1, l2)   = (1, l1) @ (l1, l2)
        """

        dL_d_left = self.right.data @ inbound_grad
        dL_d_right = inbound_grad @ self.left.data


        
        left._gradient += dL_d_left
        right._gradient += dL_d_right

'''
        
    

        
