import numpy as np
from sklearn.svm import SVR
from .utils import Decoder
__all__ = ['decoder_factory']

def decoder_factory(svr: SVR) -> Decoder:
    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        return svr.fit(X_learn.T, y_learn).predict(X_test.T)
    return _decoder

def predictor_factory(y: np.ndarray, **kwargs) -> Decoder:
    kwargs['gamma'] = kwargs.get('gamma', 1E-7)
    kwargs['epsilon'] = kwargs.get('epsilon', 1E-3)
    svr = SVR(kernel="rbf", cache_size=1E3, **kwargs)

    def _decoder(X_learn, y_learn, X_test) -> np.ndarray:
        return svr.fit(X_learn.T, y_learn).predict(X_test.T)
    return _decoder
