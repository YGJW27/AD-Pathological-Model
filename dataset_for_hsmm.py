import pandas as pd
import numpy as np

TABLE = 'D:/selected_data_ADNI.csv'
OUTPUT = 'D:/data_for_hsmm_ADNI_211212.csv'


data_table = pd.read_csv(TABLE, low_memory=False)

# attributes of interest
data_table = data_table.loc[:, ['RID','M','DX','APOE4','MMSE','LDELTOTAL','ADAS13','CDRSB']]
data_table.loc[data_table['DX'] == 'CN', 'DX'] = 1
data_table.loc[data_table['DX'] == 'MCI', 'DX'] = 2
data_table.loc[data_table['DX'] == 'Dementia', 'DX'] = 3
data_table = data_table.set_index(['RID','M'])
data_table = data_table.fillna(-1)
stacked = data_table.stack()
unstacked = stacked.unstack(1).sort_index(axis=1)
flag1 = ((~np.isnan(unstacked.iloc[:,::-1])).cumsum(axis=1) != 0).iloc[:,::-1]
flag2 = np.isnan(unstacked)
flag = flag1 & flag2
unstacked[flag] = -1

unstacked.to_csv(OUTPUT)

# stacked = data_table.stack()
# unstacked = stacked.unstack(1).sort_index(axis=1)

# expanded_time_index = np.arange(31, dtype=float)*6
# expanded_table = pd.DataFrame(index=unstacked.index, columns=expanded_time_index)
# expanded_table.columns.name = 'M'

# expanded_table.loc[:,unstacked.columns]=unstacked

# expanded_table.to_csv(OUTPUT)
