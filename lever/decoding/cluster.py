import numpy as np
from thundersvm import SVC
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from algorithm import time_series as ts
from algorithm.stats import split_data, scale_features
from algorithm.utils import quantize

PC_NO = 20

def accuracy(trials: ts.SparseRec, labels: np.ndarray, transform: str = "none",
             repeats: int = 1000, train_size: int = 30, test_size: int = 20,
             validate: bool = False, **kwargs) -> np.ndarray:
    """Give precision estimation on the estimate from a simple SVC.
    The score is calculated as tp / (tp + fp)
    where tp is true positivie and fp is false positivie.
    A 1000 time shuffle is made on the score population from different train test splits and
    the mean score is output.
    Args:
        record_file: recording file
        labels: ids of the trials belonging to different clusters
        transform: how to we get the predictors
            "none": platten neuron x sample_points and take PCs
            "mean": temporal mean so each neuron has one value per trial, then take PCs
            "corr": pairwise correlations between neurons and take PCs
        repeats: number of repeats of resampling train/test sets
    Returns:
        the distribution of mean prediction scores
    """
    X, y = trials.values, quantize(labels, groups=1)
    trial_mask = X.min(axis=2).max(axis=0) > 0
    X, y = X[:, trial_mask, :], y[trial_mask]
    X = scale_features(X, (0, 2))
    if transform == "none":
        X = np.swapaxes(X, 0, 1).reshape(X.shape[1], X.shape[0] * X.shape[2])
    elif transform == "corr":  # get inter-neuron correlation for each of the trials
        X = np.array([np.corrcoef(x)[np.triu_indices(x.shape[0], 1)] for x in np.rollaxis(X, 1)])
    elif transform == "mean":
        X = np.swapaxes(X.mean(-1), 0, 1)
    else:
        raise ValueError("[precision] <transform> must be one of 'none', 'corr', or 'mean'.")
    X = PCA(PC_NO).fit_transform(X) if X.shape[0] > PC_NO else X
    params = {"kernel": "linear", "gamma": "auto"}
    params.update(kwargs)
    svc = SVC(**params)
    splitter = split_data(X, y, repeats, train_size, test_size)
    if validate:
        result = [accuracy_score(y_te, svc.fit(X_tr, y_tr).predict(X_te)) for X_tr, y_tr, X_te, y_te in splitter]
    else:
        result = [accuracy_score(y_tr, svc.fit(X_tr, y_tr).predict(X_tr)) for X_tr, y_tr, _, _ in splitter]
    return [x for x in result if (x is not None and x > 0.0)]
