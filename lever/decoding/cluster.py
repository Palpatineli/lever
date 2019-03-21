from typing import Dict, Optional
import numpy as np
from multiprocessing import Pool

from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from noformat import File
from algorithm.array import DataFrame
from algorithm import time_series as ts

from lever.reader import load_mat

PC_NO = 20

def _process_one(split_n_classifier):
    train_X, test_X, train_y, test_y = split_n_classifier
    try:
        classifier = SVC(kernel='linear')
        classifier.fit(train_X, train_y)
    except ValueError:
        return None
    result = precision_score(test_y, classifier.predict(test_X))
    del classifier
    return result

def _get_trials(record_file: File, motion_params: Dict[str, float]) -> ts.SparseRec:
    activity = DataFrame.load(record_file["measurement"])
    lever = load_mat(record_file['response'])
    lever.center_on("motion", **motion_params)
    return ts.fold_by(activity, lever, record_file.attrs['frame_rate'], True)

def precision(record_file: File, cluster_labels: np.ndarray, motion_params: Dict[str, float],
              corr: bool = False, repeats: int = 1000, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """!!!Not Testable
    Give precision estimation on the estimate from a simple SVC.
    The score is calculated as tp / (tp + fp)
    where tp is true positivie and fp is false positivie.
    A 1000 time shuffle is made on the score population from different train test splits and
    the mean score is output.
    Args:
        record_file: recording file
        cluster_labels: ids of the trials belonging to different clusters
        motion_params: motion parameters, to extract the neuron activity surrounding the
            motion trajectory
        corr: use the correlation or not
    Returns:
        the distribution of mean prediction scores
    """
    trials = _get_trials(record_file, motion_params)
    if mask is not None:
        all_neurons, all_results = trials.values[:, mask, :], cluster_labels[mask]
    else:
        all_neurons, all_results = trials.values, cluster_labels
    if corr:
        training_X = np.array([np.corrcoef(all_neurons[:, x, :])[np.triu_indices(all_neurons.shape[0], 1)]
                               for x in range(all_neurons.shape[1])])
    else:
        training_X = np.swapaxes(all_neurons.mean(-1), 0, 1)
    training_X[np.isnan(training_X) | np.isinf(training_X)] = 0
    if training_X.shape[0] > PC_NO:
        pca_weights = PCA(PC_NO).fit_transform(training_X)
    else:
        pca_weights = training_X
    train_size, test_size = 30, 15
    if (train_size + test_size) > pca_weights.shape[0]:
        extra_number = (train_size + test_size) - pca_weights.shape[0]
        extra_idx = np.random.randint(0, pca_weights.shape[0], extra_number)
        pca_weights = np.vstack([pca_weights, pca_weights[extra_idx, :]])
        all_results = np.hstack([all_results, all_results[extra_idx]])
    splits = (train_test_split(pca_weights, all_results, train_size=train_size, test_size=test_size,
                               stratify=all_results, shuffle=True)
              for _ in range(repeats))
    print('start ', record_file.file_name)
    with Pool(processes=8) as pool:
        result = pool.map(_process_one, splits)
    print('finished')
    return [x for x in result if (x is not None and x > 0.0)]
