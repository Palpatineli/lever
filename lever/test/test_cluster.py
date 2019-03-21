from typing import List
import numpy as np

def make_null():
    return np.random.random_sample((200, 10))

def make_three_in_two():
    return np.repeat([[0, 0], [0, 5], [5, -3]], [25, 25, 50], 0) + np.random.randn(100, 2) * np.sqrt(0.05)

def make_four_in_three():
    centers: List[np.ndarray] = list()
    for _ in range(4):
        while True:
            center = np.random.randn(3) * 5
            for old_center in centers:
                if sum((old_center - center) ** 2) < 1:
                    continue
            centers.append(center)
            break
    return np.repeat(np.vstack(centers), 50, 0) + np.sqrt(0.5) * np.random.randn(200, 3)

def make_four_in_ten():
    centers: List[np.ndarray] = list()
    for _ in range(4):
        while True:
            center = np.random.randn(10) * 1.9
            for old_center in centers:
                if sum((old_center - center) ** 2) < 1:
                    continue
            centers.append(center)
            break
    return np.repeat(np.vstack(centers), 50, 0) + np.sqrt(0.5) * np.random.randn(200, 10)

def make_elongated():
    _cluster1 = np.tile(np.linspace(-0.5, 0.5, 100), (3, 1)).T
    centers = [_cluster1, _cluster1 + 10, _cluster1 + 20, _cluster1 + 15]
    return np.vstack(centers) + np.random.randn(400, 3) * 0.1
