from typing import Callable
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline
from entropy_estimators import mutual_info
from noformat import File
from lever.reader import load_mat
from .utils import Decoder

__all__ = ['decoder_power', 'cross_predict']

def hex_to_rgb(value):
    value = value.lstrip('#')
    return tuple(int(value[i:i + len(value) // 3], 16) for i in range(0, 6, 2))

def cross_predict(X: np.ndarray, y: np.ndarray, predictor: Decoder, fold: int = 10,
                  section_mi: bool = True) -> np.ndarray:
    """Calculate the prediction by taking 1 - 1 / {fold} as learning samples and predict the rest 1 / {fold} samples,
    do it {fold} times and concatenate the results.
    Args:
        X: all predictor samples, [feature x sample]
        y: [sample] real data to be predicted
        predictor: a function for prediction, takes [X_learn, y_learn, X_test] -> y_test_hat
        fold: number of folds
        section_mi: whether returns section by section mi
    """
    assert X.shape[1] == y.shape[0], f"X ({X.shape}) and y ({y.shape}) must have the sample sample size"
    length = X.shape[1]
    sections = np.c_[np.arange(fold), np.arange(1, fold + 1)] * length // fold
    end_all = sections[-1, -1]
    predicted = list()
    power = list()
    for start, end in sections:
        X_learn = np.c_[X[:, 0: start], X[:, end: end_all]]
        y_learn = np.r_[y[0: start], y[end: end_all]]
        X_test, y_test = X[:, start: end], y[start: end]
        hat = predictor(X_learn, y_learn, X_test)
        predicted.append(hat)
        if section_mi:
            power.append(mutual_info(y_test, hat))
    return ((np.hstack(predicted), np.array(power)) if section_mi else np.hstack(predicted))

def decoder_power(data_file: File, predictor_factory: Callable[[np.ndarray], Decoder]) -> float:
    lever = load_mat(data_file['response'])
    y = InterpolatedUnivariateSpline(lever.axes[0], lever.values[0])(data_file['spike']['y'])[1:]
    X = data_file['spike']['data'][:, 1:]
    decoder = predictor_factory(y)
    lever_hat, powers = cross_predict(X, y, decoder)
    return mutual_info(lever_hat, y)
