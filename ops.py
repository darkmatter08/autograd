from tensor import Tensor

from abc import ABC


class TensorOperation(ABC):
    @abstractmethod
    def forward(self):
        ...

    @abstractmethod
    def backward(self):
        ...


class MatrixMultiply(TensorOperation):
    def __init__(self, other):
        pass

    def forward(self):
        pass

    def backward(self):
        pass
