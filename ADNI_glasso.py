import pandas as pd
import numpy as np
from scipy import linalg
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLassoCV, GraphicalLasso
import sklearn.covariance

from glasso import *


CSVDATA_PATH = 'D:/Merge_SUP.csv'
ANDI_table = pd.read_csv(CSVDATA_PATH, low_memory=False)
key_var = ANDI_table.loc[:, ['AGE', 'PTEDUCAT', 'ADAS13', 'FAQ', 'Entorhinal', 'Fusiform', 'MidTemp', 'Ventricles', 'Hippocampus', 'MMSE', 'CDRSB', 'LDELTOTAL']]

key_var_drop = key_var.dropna(axis=0)

data_set = key_var_drop.to_numpy()

S = emp_covar_mat(data_set, with_std=True)

data_set -= data_set.mean(axis=0)
data_set /= data_set.std(axis=0)

emp_cov = np.dot(data_set.T, data_set) / data_set.shape[0]

covs = [('Empirical', emp_cov)]
precs = [('Empirical', linalg.inv(emp_cov))]

for alpha in [0.1, 0.3, 0.5, 0.7, 0.9]:
    model = GraphicalLasso(tol=1e-4, enet_tol=1e-4, alpha=alpha, max_iter=200).fit(data_set)
    cov_ = model.covariance_
    prec_ = model.precision_
    covs.append(('lambda={:.1f}'.format(alpha), cov_))
    precs.append(('lambda={:.1f}'.format(alpha), prec_))

vmax = cov_.max()
for i, (name, this_cov) in enumerate(covs):
    plt.subplot(2, 6, i + 1)
    plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s covariance' % name)

vmax = .9 * prec_.max()
for i, (name, this_prec) in enumerate(precs):
    ax = plt.subplot(2, 6, i + 7)
    plt.imshow(np.ma.masked_equal(this_prec, 0),
               interpolation='nearest', vmin=-vmax, vmax=vmax,
               cmap=plt.cm.RdBu_r)
    plt.xticks(())
    plt.yticks(())
    plt.title('%s precision' % name)
    if hasattr(ax, 'set_facecolor'):
        ax.set_facecolor('.7')
    else:
        ax.set_axis_bgcolor('.7')

plt.show()


G = nx.from_numpy_matrix(precs[3][1])
G = nx.relabel_nodes(G, {0: 'AGE', 1: 'PTEDUCAT', 2: 'ADAS13', 3: 'FAQ', 4: 'Entorhinal', 5: 'Fusiform', 6: 'MidTemp', 7: 'Ventricles', 8: 'Hippocampus', 9: 'MMSE', 10: 'CDRSB', 11: 'LDELTOTAL'})

nx.draw_circular(G, with_labels=True, font_weight='bold')
plt.tight_layout()
plt.show()

# model = GraphicalLassoCV(tol=1e-2, enet_tol=1e-2, max_iter=500, n_jobs=-1)
# model.fit(data_set)
# cov_ = model.covariance_
# prec_ = model.precision_

# # #############################################################################
# # Plot the results
# plt.figure(figsize=(10, 6))
# plt.subplots_adjust(left=0.02, right=0.98)

# # plot the covariances
# covs = [('Empirical', emp_cov), ('GraphicalLassoCV', cov_)]
# vmax = cov_.max()
# for i, (name, this_cov) in enumerate(covs):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(this_cov, interpolation='nearest', vmin=-vmax, vmax=vmax,
#                cmap=plt.cm.RdBu_r)
#     plt.xticks(())
#     plt.yticks(())
#     plt.title('%s covariance' % name)


# # plot the precisions
# precs = [('Empirical', linalg.inv(emp_cov)), ('GraphicalLasso', prec_)]
# vmax = .9 * prec_.max()
# for i, (name, this_prec) in enumerate(precs):
#     ax = plt.subplot(2, 2, i + 3)
#     plt.imshow(np.ma.masked_equal(this_prec, 0),
#                interpolation='nearest', vmin=-vmax, vmax=vmax,
#                cmap=plt.cm.RdBu_r)
#     plt.xticks(())
#     plt.yticks(())
#     plt.title('%s precision' % name)
#     if hasattr(ax, 'set_facecolor'):
#         ax.set_facecolor('.7')
#     else:
#         ax.set_axis_bgcolor('.7')

# # plot the model selection metric
# plt.figure(figsize=(4, 3))
# plt.axes([.2, .15, .75, .7])
# plt.plot(model.cv_results_["alphas"], model.cv_results_["mean_score"], 'o-')
# plt.axvline(model.alpha_, color='.5')
# plt.title('Model selection')
# plt.ylabel('Cross-validation score')
# plt.xlabel('alpha')

# plt.show()
# print()