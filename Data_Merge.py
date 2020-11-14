# merge ADNIMERGE and CDR key variables
import pandas as pd

Root_Path = 'D:/Projects/ADNI_Short/'
ADNIMERGE = 'Study_Info/Data___Database/ADNIMERGE.csv'
CDR = 'Assessments/Neuropsychological/CDR.csv'
OUTPUT = 'D:/Merge_SUP.csv'

merge_table = pd.read_csv(Root_Path + ADNIMERGE, low_memory=False)
cdr_table = pd.read_csv(Root_Path + CDR)
cdr_boxsum = pd.DataFrame({'CDRM_CDRSB':
    cdr_table['CDMEMORY'] + cdr_table['CDORIENT'] + cdr_table['CDJUDGE'] + 
    cdr_table['CDCOMMUN'] + cdr_table['CDHOME'] + cdr_table['CDCARE']})
cdr_memory = cdr_table.loc[:,['ID','RID','VISCODE2','EXAMDATE','CDMEMORY','CDGLOBAL']]
cdr_memory = pd.concat([cdr_memory, cdr_boxsum], axis=1)
cdr_memory.loc[cdr_memory['VISCODE2']=='sc', 'VISCODE2']='bl'
cdr_memory = cdr_memory.rename(columns={'VISCODE2': 'VISCODE', 'ID': 'CDRM_ID', 'EXAMDATE': 'CDRM_EXAMDATE'})
result = pd.merge(merge_table, cdr_memory, how='left', on=['RID', 'VISCODE'])
result = result.drop_duplicates(subset=['RID', 'VISCODE'])
result.to_csv(OUTPUT, index=False)