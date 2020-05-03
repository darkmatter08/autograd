import numpy as np

from .tensor import Tensor


def test_matmul_data():
    t1 = Tensor(np.array([[1, 1, 1]]))
    t2 = Tensor(np.array([[2], [2], [2]]))

    t = t1 @ t2
    answer = np.array([[6]])

    assert t.shape == answer.shape
    assert (t.data == answer).all()


def test_matmul_scalar_grads():
    s1 = Tensor(np.array([[2.]]))
    s2 = Tensor(np.array([[3.]]))

    s = s1 @ s2
    s.backward(np.array([[1.]]))

    assert (s1._gradient == np.array([[3.]])).all()
    assert (s2._gradient == np.array([[2.]])).all()
    

'''
def test_matmul_grads():
    t1 = Tensor(np.array([[1., 1., 1.]]))
    t2 = Tensor(np.array([[2.], [2.], [2.]]))

    t = t1 @ t2
    answer = np.array([[6]])

    
    assert t.shape == answer.shape
    assert (t.data == answer).all()
'''    

    
