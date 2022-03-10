import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, rv_continuous
import seaborn as sns


def logit(x):
    y = np.log(x/(1-x))
    return y


class LogitNormal(rv_continuous):
    def __init__(self, scale=1, loc=0):
        super().__init__(self)
        self.scale = scale
        self.loc = loc

    def _pdf(self, x):
        return norm.pdf(logit(x), loc=self.loc, scale=self.scale)/(x*(1-x))


def skewness(data):
    numer = np.sum(np.power(data - np.mean(data), 3)) / data.shape[0]
    denom = np.power(np.sqrt(np.sum(np.power(data - np.mean(data), 2)) / data.shape[0]), 3)
    return numer/denom

DATA_SOURCE = 'D:/data_for_hsmm_ADNI_211212_origin.csv'

data_table = pd.read_csv(DATA_SOURCE, low_memory=False)

# Distinguish between normal aging progression and AD progression
data_table['DX'] = data_table['DX'].fillna(method='ffill')      # fill na in DX
subject_id = data_table.loc[:, 'RID']
id_set = subject_id.unique()
id_to_delete = np.zeros_like(id_set)
for idx, id in enumerate(id_set):
    id_table = data_table.loc[data_table.loc[:, 'RID'] == id, :]
    if (np.any(id_table['DX'] == 1) & ~np.any(id_table['DX'] == 2) & ~np.any(id_table['DX'] == 3)):
        id_to_delete[idx] = 1

# delete normal aging progression data
data_table_without_CNP = data_table[~data_table['RID'].isin(id_set[id_to_delete == 1])]


# -------------------- calculate the distribution of observation in each state -------------------- #
path = 'D:/Project/AD_progression/'
# observation in CN state
obs = (data_table_without_CNP[data_table_without_CNP['DX'] == 1]['MMSE'] / 30) * 0.9 + 0.05
print(skewness(obs))
plt.hist(obs, density=True, color = 'blue', edgecolor = 'black', bins = (np.arange(30)-0.5)/30)
plt.xlim(-0.2, 1.2)
mu, std = norm.fit(obs.dropna())
x = np.linspace(-0.2, 1.2, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k--', linewidth=2, label='Normal distribution')

data = logit(obs.dropna())
print(skewness(data))
mu, std = norm.fit(data)
values = np.linspace(10e-10, 1-10e-10, 1000)
plt.plot(values, LogitNormal(loc=mu, scale=std).pdf(values), 'r-', linewidth=2, label='Logit-Normal distribution')
plt.legend(loc='upper left')
plt.savefig(path + 'MMSE_CN.png')
plt.show()
plt.clf()


# observation in MCI state
obs = (data_table_without_CNP[data_table_without_CNP['DX'] == 2]['MMSE'] / 30) * 0.9 + 0.05
print(skewness(obs))
plt.hist(obs, density=True, color = 'blue', edgecolor = 'black', bins = (np.arange(30)-0.5)/30)
plt.xlim(-0.2, 1.2)
mu, std = norm.fit(obs.dropna())
x = np.linspace(-0.2, 1.2, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k--', linewidth=2, label='Normal distribution')

data = logit(obs.dropna())
print(skewness(data))
mu, std = norm.fit(data)
values = np.linspace(10e-10, 1-10e-10, 1000)
plt.plot(values, LogitNormal(loc=mu, scale=std).pdf(values), 'r-', linewidth=2, label='Logit-Normal distribution')
plt.legend(loc='upper left')
plt.savefig(path + 'MMSE_MCI.png')
plt.show()
plt.clf()

# observation in AD state
obs = (data_table_without_CNP[data_table_without_CNP['DX'] == 3]['MMSE'] / 30) * 0.9 + 0.05
print(skewness(obs))
plt.hist(obs, density=True, color = 'blue', edgecolor = 'black', bins = (np.arange(30)-0.5)/30)
plt.xlim(-0.2, 1.2)
mu, std = norm.fit(obs.dropna())
x = np.linspace(-0.2, 1.2, 100)
p = norm.pdf(x, mu, std)
plt.plot(x, p, 'k--', linewidth=2, label='Normal distribution')

data = logit(obs.dropna())
print(skewness(data))
mu, std = norm.fit(data)
values = np.linspace(10e-10, 1-10e-10, 1000)
plt.plot(values, LogitNormal(loc=mu, scale=std).pdf(values), 'r-', linewidth=2, label='Logit-Normal distribution')
plt.legend(loc='upper left')
plt.savefig(path + 'MMSE_AD.png')
plt.show()
plt.clf()