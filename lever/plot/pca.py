from typing import Optional
import numpy as np
from numpy.linalg import lstsq
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from sklearn.decomposition import PCA

def draw_pc(X, y, ax: Optional[Axes] = None, **kwargs):
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots(111, projection="3d")
    pc3 = PCA(3).fit_transoform(X)
    ax.scatter(pc3[:, 0], pc3[:, 1], pc3[:, 2], c=y, **kwargs)

def draw_decision_plane(X: np.ndarray, y: np.ndarray, plane_coef: np.ndarray, ax: Optional[Axes] = None,
                        grid_no: int = 20, **kwargs):
    """Draw pc in 3d for data points, and a decision plane given in coefficients.
    Args:
        X: n_sample x n_feature
        y: label, a 1D n_sample array
        plane_coef: 1D n_feature + 1 array, last dimension being intercept
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.subplots(111, projection="3d")
    pca = PCA(3).fit(X)
    pc3 = pca.transform(X)
    ax.scatter(pc3[:, 0], pc3[:, 1], pc3[:, 2], c=y)
    pc_coef = lstsq(pca.components_.T, plane_coef[:-1].ravel(), rcond=None)[0]
    xx, yy = np.meshgrid(np.linspace(*ax.get_xlim(), grid_no), np.linspace(*ax.get_ylim(), grid_no))
    zz = (xx * pc_coef[0] + yy * pc_coef[1] + plane_coef[-1]) / (-pc_coef[2])
    zlims = ax.get_zlim()
    ax.plot_surface(xx, yy, zz, alpha=0.5)
    ax.set_zlim(zlims)
