from lever.plot.pca import draw_decision_plane
from lever.script.steps.trial_neuron import res_trial_neuron
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
clu = res_cluster.run(*param_dict[22])
y = quantize(clu)[:-1]
X = res_trial_neuron.run(*param_dict[22]).values
X = np.swapaxes(X, 0, 1).reshape(X.shape[1], -1)[:-1, :]
new_X = PCA(10).fit_transform(X)
svc = SVC(kernel='linear', gamma='auto').fit(new_X, y)
decision_plane = np.hstack([svc.coef_[0], 0])
draw_decision_plane(new_X, y, decision_plane, cmap='viridis')

