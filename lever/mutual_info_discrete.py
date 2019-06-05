from typing import Callable, Dict
import numpy as np
from sklearn.metrics import mutual_info_score

def mutual_info(x: np.ndarray, y: np.ndarray, style: str = "freedman") -> float:
    """calculate mutual information by binning samples
    Calculate the bin for distribution histogram by freedman-diaconis rule: http://bit.ly/2Gu1Kyu
    $D_x = (max(X) - min(X)) / (2 IQR n^{-1/3})$
    Args:
        style: "freedman", "scott", or "sturges"
    """
    def freedman(x: np.ndarray) -> int:
        iqr = np.quantile(x, [0.25, 0.75])
        return int((x.max() - x.min()) / (2 * (iqr[1] - iqr[0]) * x.size ** (-1 / 3))) + 1

    def scott(x: np.ndarray) -> int:
        return int((x.max() - x.min()) / (3.5 * np.std(x) * x.size ** (-1 / 3))) + 1

    def sturges(x: np.ndarray) -> int:
        return int(1 + np.log2(x.size)) + 1

    bin_funcs: Dict[str, Callable[[np.ndarray], float]] = {"freedman": freedman, "scott": scott, "sturges": sturges}
    if style in bin_funcs:
        bin_func = bin_funcs[style]
        bin_no = [bin_func(x), bin_func(y)]
    else:
        try:
            bin_no = [int(style), int(style)]
        except ValueError:
            raise ValueError("style can only be one of ['freedman', 'scott', 'sturges']")
    dist = np.histogram2d(x, y, bin_no)[0]
    return mutual_info_score(None, None, contingency=dist)

def aic(x, y, bin_no):
    dist = np.histogram2d(x, y, bin_no)[0]
    j = np.count_nonzero(dist)
    return (j - mutual_info_score(None, None, contingency=dist) * x.size) * 2
