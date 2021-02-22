import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.covariance import GraphicalLasso
import sklearn.covariance

from glasso import *

# root_path = "D:\\Project\\Sachs.SOM.Datasets\\Data Files\\"

# csv_name = ["1. cd3cd28.xls",
#             "2. cd3cd28icam2.xls",
#             "3. cd3cd28+aktinhib.xls",
#             "4. cd3cd28+g0076.xls",
#             "5. cd3cd28+psitect.xls",
#             "6. cd3cd28+u0126.xls",
#             "7. cd3cd28+ly.xls",
#             "8. pma.xls",
#             "9. b2camp.xls"]

# data = []
# for name in csv_name:
#     df = pd.read_excel(root_path+name)
#     data.append(df.to_numpy())

# data_set = data[0]
# for i in range(1, len(data)):
#     data_set = np.concatenate((data_set, data[i]), axis=0)

root_path = "D:\\\Project\\Sachs.SOM.Datasets\\sachsCtsHTF.txt"
df = pd.read_csv(root_path, sep=' ', header=None)
data_set = df.to_numpy()



S = emp_covar_mat(data_set, with_std=True)

cov = GraphicalLasso(alpha=36).fit(data_set)


W, theta = sklearn.covariance.graphical_lasso(emp_cov=np.array(S), alpha=50)


# W, theta, (m, dW) = ggmlasso(S, lam=36)


print("W:", W)
print("theta:", theta)
# print("m:", m)
# print("dW", dW)


# G = nx.from_numpy_matrix(theta)
# G = nx.relabel_nodes(G, {0: 'Raf', 1: 'Mek', 2: 'Plcg', 3: 'PIP2', 4: 'PIP3', 5: 'Erk', 6: 'Akt', 7: 'PKA', 8: 'PKC', 9: 'P38', 10: 'Jnk'})

# nx.draw(G, with_labels=True, font_weight='bold')
# plt.show()