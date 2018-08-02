from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import gc
import sys
import warnings
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
nrows=None
#nrows=1000
data = pd.read_csv('../output/data_eng.csv',nrows=nrows)
test = pd.read_csv('../output/test_eng.csv',nrows=nrows)
#data.info(verbose=True)
#sys.exit(0)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # 0 !!!!!
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])
#import catboost
y = data['TARGET']
del data['TARGET']
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
CATEGORICAL_COLUMNS = ['CODE_GENDER',
                       'EMERGENCYSTATE_MODE',
                       'FLAG_CONT_MOBILE',
                       'FLAG_DOCUMENT_3',
                       'FLAG_DOCUMENT_4',
                       'FLAG_DOCUMENT_5',
                       'FLAG_DOCUMENT_6',
                       'FLAG_DOCUMENT_7',
                       'FLAG_DOCUMENT_8',
                       'FLAG_DOCUMENT_9',
                       'FLAG_DOCUMENT_11',
                       'FLAG_DOCUMENT_18',
                       'FLAG_EMAIL',
                       'FLAG_EMP_PHONE',
                       'FLAG_MOBIL',
                       'FLAG_OWN_CAR',
                       'FLAG_OWN_REALTY',
                       'FLAG_PHONE',
                       'FLAG_WORK_PHONE',
                       'FONDKAPREMONT_MODE',
                       'HOUR_APPR_PROCESS_START',
                       'HOUSETYPE_MODE',
                       'LIVE_CITY_NOT_WORK_CITY',
                       'LIVE_REGION_NOT_WORK_REGION',
                       'NAME_CONTRACT_TYPE',
                       'NAME_TYPE_SUITE',
                       'NAME_INCOME_TYPE',
                       'NAME_EDUCATION_TYPE',
                       'NAME_FAMILY_STATUS',
                       'NAME_HOUSING_TYPE',
                       'OCCUPATION_TYPE',
                       'ORGANIZATION_TYPE',
                       'REG_CITY_NOT_LIVE_CITY',
                       'REG_CITY_NOT_WORK_CITY',
                       'REG_REGION_NOT_LIVE_REGION',
                       'REG_REGION_NOT_WORK_REGION',
                       'WALLSMATERIAL_MODE',
                       'WEEKDAY_APPR_PROCESS_START']
CATEGORICAL_COLUMNS=[c for c in CATEGORICAL_COLUMNS if c in data.columns.tolist()]
print(CATEGORICAL_COLUMNS)
cl=[feats.index(c) for c in CATEGORICAL_COLUMNS]
CATEGORICAL_COLUMNS=cl
print(CATEGORICAL_COLUMNS)
#`import catboost
#catboost.__version__
from catboost import CatBoostClassifier
#import catboost as cb
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data,y)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    model = CatBoostClassifier(random_seed=42,iterations=5000,metric_period=100,
                               bagging_temperature=0.5,
                               depth=4, l2_leaf_reg=5, learning_rate=0.07,eval_metric='AUC',od_type="Iter",od_wait=300)
    print('Fold '+str(n_fold+1))
    model.fit(trn_x, trn_y,eval_set=(val_x, val_y),
	      use_best_model=True,  verbose=200,cat_features=CATEGORICAL_COLUMNS)
    val_pred=model.predict_proba(val_x)
    oof_preds[val_idx] = model.predict_proba(val_x)[:, 1]
    test_pred=model.predict_proba(test[feats])
    sub_preds += test_pred[:, 1] / folds.n_splits
    print("roc_auc_score = {}".format(roc_auc_score(val_y, val_pred[:, 1])))
    del model, trn_x, trn_y, val_x, val_y,val_pred,test_pred
    gc.collect()
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))