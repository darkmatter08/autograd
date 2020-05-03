from tensor import Tensor

import abc


class TensorOperation(abc.ABC):
    @abc.abstractmethod
    def forward(self):
        ...

    @abc.abstractmethod
    def backward(self):
        ...


class MatrixMultiply(TensorOperation):
    def __init__(self, other):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
