import numpy as np

from .tensor import Tensor



def test_tensor_default():
    t = Tensor(np.array([]))


def test_tensor_with_data():
    t = Tensor(np.array([1, 2, 3]))

def test_tensor_with_dependencies():
    t1 = Tensor(np.array([1, 1, 1]))
    t = Tensor(np.array([1, 2, 3]), t1)

