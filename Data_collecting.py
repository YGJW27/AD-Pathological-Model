# select key attributes from merge table, and filter out short term subjects
import pandas as pd

Root_Path = 'D:/Project/ADNI_Short/Study_Info/'
ADNIMERGE = 'Data___Database/ADNIMERGE.csv'
OUTPUT = 'D:/selected_data_ADNI.csv'

merge_table = pd.read_csv(Root_Path + ADNIMERGE, low_memory=False)
selected = merge_table.loc[:,[
    'RID','PTID','VISCODE','M','DX_bl','DX','AGE','PTGENDER','PTEDUCAT','APOE4',
    'CDRSB','ADAS11','ADAS13','MMSE','RAVLT_immediate','RAVLT_learning',
    'RAVLT_forgetting','RAVLT_perc_forgetting','LDELTOTAL','DIGITSCOR','TRABSCOR','FAQ']]

# delete error and duplicated rows
selected = selected[selected.VISCODE != 'y1']
selected = selected[selected.M != 3]
selected = selected.drop_duplicates(subset=['RID', 'M'])

id_iter = selected[selected.duplicated(subset=['RID'])==False].RID

# Filter out subjects with examine less than 6 months
thres = 36
for id in id_iter:
    long_flag = (selected[selected.RID == id].M >= thres).any()
    if long_flag == False:
        selected = selected[selected.RID != id]

selected = selected.sort_values(by=['RID','M'])

selected.to_csv(OUTPUT, index=False)