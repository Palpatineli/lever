import numpy as np
from lever.decoding.utils import split_by_proportion, oversample

def test_split_by_proportion():
    state = np.random.RandomState(12345)
    proportions = state.rand(10)
    proportions /= proportions.sum()
    result = split_by_proportion(proportions, 50)
    assert np.array_equal(result, np.array([8, 3, 1, 2, 5, 5, 8, 6, 6, 6]))

def test_oversample():
    state = np.random.RandomState(12345)
    X = np.arange(20).reshape(5, 4)
    y = state.randint(0, 3, 5)
    Xa, ya = oversample(X, y, 15)
    assert(ya.shape[0] == 15)
    assert(np.mean(ya == 2) == np.mean(y == 2))
    y = state.randint(0, 2, 5)
    Xa, ya = oversample(X, y, 15)
    assert(np.abs(np.mean(y == 1) - np.mean(y == 1)) < 0.03)
