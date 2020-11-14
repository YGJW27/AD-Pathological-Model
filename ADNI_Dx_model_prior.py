import pandas as pd
import sklearn.tree as tree
import matplotlib.pyplot as plt
import graphviz
from sklearn.metrics import classification_report


CSVDATA_PATH = 'D:/Merge_SUP.csv'

ANDI_table = pd.read_csv(CSVDATA_PATH, low_memory=False)
key_var = ANDI_table.loc[:, ['RID', 'VISCODE', 'AGE', 'PTEDUCAT', 'MMSE', 'LDELTOTAL', 'CDMEMORY', 'CDGLOBAL', 'DX']]
keep_3_status_table = key_var[(key_var['DX'] == 'CN') | (key_var['DX'] == 'MCI') | (key_var['DX'] == 'Dementia')]
label = keep_3_status_table['DX']
label[label == 'CN'] = 0
label[label == 'MCI'] = 1
label[label == 'Dementia'] = 2
label = label.astype('int')
X = keep_3_status_table.loc[:, ['PTEDUCAT', 'MMSE', 'LDELTOTAL', 'CDMEMORY', 'CDGLOBAL']]

# prior
X.loc[X['PTEDUCAT'] >= 16, 'PTEDUCAT'] = 16
X.loc[X['PTEDUCAT'] <= 7, 'PTEDUCAT'] = 7
X.loc[X['CDGLOBAL'] >= 1, 'CDGLOBAL'] = 1
X.loc[X['CDMEMORY'] >= 1, 'CDMEMORY'] = 1
X.loc[X['MMSE'] <= 15, 'MMSE'] = 15
X.loc[X['LDELTOTAL'] <= 2, 'LDELTOTAL'] = 2
X.loc[X['LDELTOTAL'] >= 9, 'LDELTOTAL'] = 9

MMSE_miss = keep_3_status_table['MMSE'].isna()
LDEL_miss = keep_3_status_table['LDELTOTAL'].isna()
MEMO_miss = (keep_3_status_table['CDMEMORY'].isna()) | (keep_3_status_table['CDMEMORY'] == -1)
CDR_miss = (keep_3_status_table['CDGLOBAL'].isna()) | (keep_3_status_table['CDGLOBAL'] == -1)

most_one_missing =  (~MMSE_miss & ~LDEL_miss & ~MEMO_miss & ~CDR_miss)| \
                    (MMSE_miss & ~LDEL_miss & ~MEMO_miss & ~CDR_miss) | \
                    (~MMSE_miss & LDEL_miss & ~MEMO_miss & ~CDR_miss) | \
                    (~MMSE_miss & ~LDEL_miss & MEMO_miss & ~CDR_miss) | \
                    (~MMSE_miss & ~LDEL_miss & ~MEMO_miss & CDR_miss)
no_missing = ~MMSE_miss & ~LDEL_miss & ~MEMO_miss & ~CDR_miss

X_most_one_missing = X[most_one_missing]
label_most_one_missing = X[most_one_missing]
X_no_missing = X[no_missing]
label_no_missing = label[no_missing]

# train a decision tree classifier
clf = tree.DecisionTreeClassifier(max_leaf_nodes=100)
clf = clf.fit(X_no_missing, label_no_missing)
pred_y = clf.predict(X_no_missing)
print(classification_report(label_no_missing, pred_y, target_names=['CN', 'MCI', 'Dementia']))
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=X.columns, class_names=['CN', 'MCI', 'Dementia'], filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render('D:/AD_tree_prior_100')
