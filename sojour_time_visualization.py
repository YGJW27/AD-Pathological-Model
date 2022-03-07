import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def sojour_time_func(df, state):
    sojour_time_list = np.array([])
    df = df.sort_values(by='M')
    time_start = 0
    start_flag = 0
    for _, row in df.iterrows():
        if (row['DX'] == state) & ~start_flag:
            time_start = row['M']
            start_flag = 1
        elif (row['DX'] != state) & start_flag:
            start_flag = 0
            sojour_time = row['M'] - time_start
            sojour_time_list = np.append(sojour_time_list, sojour_time)
    return sojour_time_list
        

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


# -------------------- calculate the distribution of sojour time -------------------- #
# sojour time distribution of CN
subject_id = data_table_without_CNP.loc[:, 'RID']
id_set = subject_id.unique()
sojour_time_list = np.array([])
for id in id_set:
    id_table = data_table.loc[data_table.loc[:, 'RID'] == id, :]
    sojour_time_set = sojour_time_func(id_table, 1)
    for sojour_time in sojour_time_set:
        sojour_time_list = np.append(sojour_time_list, sojour_time)
CN_sojour_count = sojour_time_list
print()

# sojour time distribution of MCI
subject_id = data_table_without_CNP.loc[:, 'RID']
id_set = subject_id.unique()
sojour_time_list = np.array([])
for id in id_set:
    id_table = data_table.loc[data_table.loc[:, 'RID'] == id, :]
    sojour_time_set = sojour_time_func(id_table, 2)
    for sojour_time in sojour_time_set:
        sojour_time_list = np.append(sojour_time_list, sojour_time)
MCI_sojour_count = sojour_time_list

# draw the histogram of sojour time
path = 'D:/Project/AD_progression/'
plt.hist(CN_sojour_count, color = 'blue', edgecolor = 'black',
         bins = int(180/12))
plt.savefig(path + 'CN_histogram.png')
plt.show()
plt.clf()

plt.hist(MCI_sojour_count, color = 'blue', edgecolor = 'black',
         bins = int(180/12))
plt.savefig(path + 'MCI_histogram.png')
plt.show()
plt.clf()
print()