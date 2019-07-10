##
from typing import List
from pathlib import Path
from multiprocessing import Pool, cpu_count
import numpy as np
from matplotlib import pyplot as plt
from statsmodels import api as sm
from pypedream import Task, getLogger
from encoding_model.main import build_poly_predictor, _corr, scale
from encoding_model.utils import split_folds
from lever.script.steps.encoding import res_predictor
from lever.script.steps.utils import read_index

proj_folder = Path.home().joinpath("Sync/project/2018-leverpush-chloe/")
Task.save_folder = proj_folder.joinpath("data", "interim")
mice = read_index(proj_folder)

def test_model():
    log = getLogger("model-test", "model-test.log")
    param_dict = [(item.name, log) for item in mice]
    pool = Pool(max(1, cpu_count() - 2))
    result = pool.starmap(res_predictor.run, param_dict[0: 1])
    X, y, grouping = result[0]
    flat_X, grouping = build_poly_predictor(X, y, grouping)
    flat_y = y.reshape(y.shape[0], -1)
    flat_y = flat_y - flat_y.min(axis=0, keepdims=True) + 1E-6
    # flat_y = scale(y.reshape(y.shape[0], -1), 1)
    # glm_family = sm.families.Gaussian(sm.families.links.identity())
    glm_families = {"possion-log": sm.families.Poisson(sm.families.links.log()),
                    "gaussian": sm.families.Gaussian(sm.families.links.identity())}
    alpha_list = np.linspace(0, 0.4, 16)
    for name, glm_family in glm_families.items():
        r2s = list()
        for alpha in alpha_list:
            r2s.append(validated_r2(flat_X, flat_y, alpha, glm_family))
        array = np.vstack([alpha_list, np.array(r2s).T])
        np.savetxt(proj_folder.joinpath("data", "testing", f"glm-{name}.csv"), array, delimiter=",")
        plt.plot(np.linspace(0, 0.4, 16), np.nanmean(r2s, axis=1), label=name)

##
def validated_r2(X, y, alpha, glm_family, folds: int = 5) -> List[float]:
    r2s = list()
    for neuron in y:
        y_hat = np.zeros(neuron.shape[0] // folds * folds, dtype=neuron.dtype)
        for idx_tr, idx_te in split_folds(neuron.shape[0], folds):
            temp_X = sm.add_constant(X)
            model = sm.GLM(neuron[idx_tr], temp_X[idx_tr, :], family=glm_family).fit_regularized(alpha=alpha)
            y_hat[idx_te] = model.predict(temp_X[idx_te])
        r2s.append(_corr(y_hat, y[:y_hat.shape[0]]))
    return r2s

##
