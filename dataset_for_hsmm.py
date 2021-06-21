import pandas as pd
import numpy as np

TABLE = 'D:/selected_data_ADNI.csv'
OUTPUT = 'D:/data_for_hsmm_ADNI.csv'


data_table = pd.read_csv(TABLE, low_memory=False)

# attributes of interest
data_table = data_table.loc[:, ['RID','M','MMSE','LDELTOTAL']]
data_table = data_table.set_index(['RID','M'])
stacked = data_table.stack()
unstacked = stacked.unstack(1).sort_index(axis=1)

expanded_time_index = np.arange(31, dtype=float)*6
expanded_table = pd.DataFrame(index=unstacked.index, columns=expanded_time_index)
expanded_table.columns.name = 'M'

expanded_table.loc[:,unstacked.columns]=unstacked

expanded_table.to_csv(OUTPUT)
