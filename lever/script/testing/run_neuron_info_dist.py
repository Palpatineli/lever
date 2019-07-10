##
from typing import Tuple, Callable
from os.path import expanduser, join
import pickle as pkl
import numpy as np
from scipy.stats import linregress, chi2, ttest_ind
from scipy.optimize import minimize, OptimizeResult
from scipy.special import polygamma
import matplotlib.pyplot as plt
from algorithm.utils import map_tree
from algorithm.stats import combine_test, perm_test
from mplplot import Figure
from lever.plot import plot_scatter

COLORS = ["#dc322fff", "#268bd2ff", "#d33682ff", "#2aa198ff", "#859900ff", "#b58900ff"]
project_folder = expanduser("~/Sync/project/2018-leverpush-chloe")
img_folder = join(project_folder, 'report', 'img')
res_folder = join(project_folder, 'report', 'measure')
##
with open(join(res_folder, "svr_power.pkl"), 'rb') as fp:
    result = pkl.load(fp)
ind_scores = {x: [a[1] for a in y] for x, y in result.items()}
wt_size = [len(x) for x in ind_scores['wt']]
glt_size = [len(x) for x in ind_scores['glt1']]
dredd_size = [len(x) for x in ind_scores['dredd']]
print(f"wt: {np.mean(wt_size)}, glt: {np.mean(glt_size)}, dredd: {np.mean(dredd_size)}")
##
score_no = map_tree(lambda x: len(x), ind_scores)
plt.hist(score_no.values(), 50)
## Test: does number of neurons affect slope?
pool = np.exp(-np.arange(250) / 25)
def take_sample(cell_no: int, pool: np.ndarray, fn):
    res = list()
    for _ in range(500):
        samples = list()
        x_axis = list()
        for _ in range(10):
            sample = np.flip(np.sort(np.random.choice(pool, cell_no, replace=False)))
            sample /= np.sum(sample)
            samples.append(sample)
            x_axis.append(np.linspace(1, 61, cell_no))
        sample_stack = np.hstack(samples)
        x_stack = np.hstack(x_axis)
        res.append(fn(x_stack, sample_stack))
    print('done')
    return res
## shit, log linregress is totally dependent on no
res40 = [x.slope for x in take_sample(40, pool, lambda x, y: linregress(x, np.log(y)))]
res50 = [x.slope for x in take_sample(50, pool, lambda x, y: linregress(x, np.log(y)))]
res60 = [x.slope for x in take_sample(60, pool, lambda x, y: linregress(x, np.log(y)))]
print(f"res40: {np.mean(res40)}, res50: {np.mean(res50)}, res60: {np.mean(res60)}")
## chisq fit
def chi2fn(θ: Tuple[float, float, float], x, y) -> Tuple[float, np.ndarray]:
    """Let y_hat = A * chi2(x, γ). return mean sq diff
    Args:
        θ: (A, γ)
    """
    a, γ, μ = θ
    ξ = chi2.pdf(x / μ, df=γ)
    value = np.mean((y - a * ξ) ** 2)
    da = -np.mean(2 * (y - a * ξ) * ξ)
    dξ_dγ = -ξ / 2 * ((np.log(2 * μ / x) + polygamma(0, γ / 2)))
    dγ = np.mean(2 * (y - a * ξ) * a * ξ * dξ_dγ)
    dξ_dμ = (x / (2 * μ) - γ / 2 + 1) * ξ
    dμ = np.mean(2 * (y - a * ξ) * a * ξ * dξ_dμ)
    return value, np.array([da, dγ, dμ])

def get_chi(x, y) -> OptimizeResult:
    return minimize(chi2fn, (1.0, 1.0, 5.0), (x, y), "L-BFGS-B", jac=True,
                    bounds=((0.001, 10.), (0.0001, 10.0), (None, None)))
##
def expfn(θ: Tuple[float, float], x, y) -> Tuple[float, np.ndarray]:
    """Let y_hat = A * exp(-x / τ). return mean sq diff
    Args:
        θ: (A, τ)
    """
    a, τ = θ
    da = np.exp(-x / τ)
    y_hat = a * da
    dτ = y_hat * x / τ ** 2
    return np.mean((y - y_hat) ** 2), np.mean(np.asarray([da, dτ]) * (-2 * (y - y_hat)), axis=1)

def get_exp(x, y) -> OptimizeResult:
    return minimize(expfn, (0.2, 1), (x, y), "L-BFGS-B", jac=True)
## show fit
def show_fit(cell_no: int, pool: np.ndarray, fit_fn: Callable[[np.ndarray, np.ndarray], OptimizeResult],
             fn: Callable[..., np.ndarray]):
    sample = np.flip(np.sort(np.random.choice(pool, cell_no, replace=False)))
    x, x_axis = np.arange(1, cell_no + 1), np.linspace(1, cell_no + 1, 200)
    result = fit_fn(x, sample)
    y_hat = fn(result.x, x_axis)
    plt.plot(x_axis, y_hat)
    plt.scatter(np.arange(1, 41), sample, color='green')
##
show_fit(40, pool, get_chi, lambda θ, x: θ[0] * chi2.pdf(x / θ[2], df=θ[1]))
##
res = dict()
for count in (40, 50, 60):
    res[count] = take_sample(count, pool, lambda x, y: get_chi(x, y).x[1])
print(f"res40: {np.mean(res[40])}, res50: {np.mean(res[50])}, res60: {np.mean(res[60])}")
##
wt_0 = ind_scores['wt'][0]
with Figure() as (ax,):
    for color, (group_str, group) in zip(COLORS, ind_scores.items()):
        for case in group:
            cumsum = np.cumsum(np.flip(np.sort(case)))
            cumsum /= cumsum[-1]
            cumsum = np.hstack([[0], cumsum])
            ax.plot(np.linspace(0, 1, cumsum.shape[0]), cumsum, color=color)
##
def get_roi(x, repeat=1000):
    res = list()
    for _ in range(repeat):
        cumsum = np.cumsum(np.flip(np.sort(np.random.choice(x, 25, True))))
        cumsum /= cumsum[-1]
        cumsum = np.hstack([[0], cumsum])
        res.append(((cumsum.sum() - cumsum[-1] / 2) / cumsum.shape)[0])
    return np.mean(res)

result = map_tree(get_roi, ind_scores)
del result['glt1'][10]
del result['wt'][7]
print(combine_test(result, [ttest_ind, perm_test]))
plot_scatter(result, COLORS)
##
perm_test(result['glt1'], result['dredd'], 10000)
##
def sampling(cell_no, pool) -> np.ndarray:
    res = list()
    for _ in range(100):
        sample = np.random.choice(pool, cell_no)
        for _ in range(100):
            res.append(get_roi(np.random.choice(sample, 20)))
    return np.asarray(res)
##
pool = np.exp(-np.arange(250) / 25)
res = sampling(40, pool)
print(f"40: {res.mean()}, {res.std()}")
res = sampling(50, pool)
print(f"50: {res.mean()}, {res.std()}")
res = sampling(60, pool)
print(f"60: {res.mean()}, {res.std()}")
##
pool0 = np.exp(-np.arange(250) / 25)
res = sampling(40, pool0)
print(f"25: {res.mean()}, {res.std()}")
pool1 = np.exp(-np.arange(250) / 50)
res = sampling(40, pool1)
print(f"50: {res.mean()}, {res.std()}")
##
##
