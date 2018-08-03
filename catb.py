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
from time import gmtime, strftime,time
nrows=None
#nrows=1000
print("Loading data")
data = pd.read_csv('../output/data_eng.csv',nrows=nrows)
test = pd.read_csv('../output/test_eng.csv',nrows=nrows)
#data.info(verbose=True)
#sys.exit(0)
AGGREGATION_RECIPIES = [
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE'], [('AMT_ANNUITY', 'max'),
                                              ('AMT_CREDIT', 'max'),
                                              ('EXT_SOURCE_1', 'mean'),
                                              ('EXT_SOURCE_2', 'mean'),
                                              ('OWN_CAR_AGE', 'max'),
                                              ('OWN_CAR_AGE', 'sum')]),
    (['CODE_GENDER', 'ORGANIZATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                                            ('AMT_INCOME_TOTAL', 'mean'),
                                            ('DAYS_REGISTRATION', 'mean'),
                                            ('EXT_SOURCE_1', 'mean')]),
    (['CODE_GENDER', 'REG_CITY_NOT_WORK_CITY'], [('AMT_ANNUITY', 'mean'),
                                                 ('CNT_CHILDREN', 'mean'),
                                                 ('DAYS_ID_PUBLISH', 'mean')]),
    (['CODE_GENDER', 'NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('EXT_SOURCE_1', 'mean'),
                                                                                           ('EXT_SOURCE_2', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE'], [('AMT_CREDIT', 'mean'),
                                                  ('AMT_REQ_CREDIT_BUREAU_YEAR', 'mean'),
                                                  ('APARTMENTS_AVG', 'mean'),
                                                  ('BASEMENTAREA_AVG', 'mean'),
                                                  ('EXT_SOURCE_1', 'mean'),
                                                  ('EXT_SOURCE_2', 'mean'),
                                                  ('EXT_SOURCE_3', 'mean'),
                                                  ('NONLIVINGAREA_AVG', 'mean'),
                                                  ('OWN_CAR_AGE', 'mean'),
                                                  ('annuity_income_percentage', 'mean'),
                                                  ('YEARS_BUILD_AVG', 'mean')]),
    (['NAME_EDUCATION_TYPE', 'OCCUPATION_TYPE', 'REG_CITY_NOT_WORK_CITY'], [('ELEVATORS_AVG', 'mean'),
                                                                            ('EXT_SOURCE_1', 'mean')]),
    (['OCCUPATION_TYPE'], [('AMT_ANNUITY', 'mean'),
                           ('CNT_CHILDREN', 'mean'),
                           ('CNT_FAM_MEMBERS', 'mean'),
                           ('DAYS_BIRTH', 'mean'),
                           ('DAYS_EMPLOYED', 'mean'),
                           ('DAYS_ID_PUBLISH', 'mean'),
                           ('DAYS_REGISTRATION', 'mean'),
                           ('EXT_SOURCE_1', 'mean'),
                           ('EXT_SOURCE_2', 'mean'),
                           ('EXT_SOURCE_3', 'mean')]),]
groupby_aggregate_names = []
print('Shapes : ', data.shape, test.shape)
for groupby_cols, specs in AGGREGATION_RECIPIES:
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        groupby_aggregate_names.append(groupby_aggregate_name)
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
#import catboost
feats = [f for f in data.columns if f not in ['SK_ID_CURR','TARGET']]
#print(CATEGORICAL_COLUMNS)
#`import catboost
#catboost.__version__
from catboost import CatBoostClassifier
#import catboost as cb
dr_feat=[c for c in groupby_aggregate_names if c in data.columns.tolist()]
data.drop(dr_feat,axis=1,inplace=True)
test.drop(dr_feat,axis=1,inplace=True)
#feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
#data.drop(feats[1134],axis=1,inplace=True)
#test.drop(feats[1134],axis=1,inplace=True)
#data=data[~data[feats[1134]].isnull()]
#data=data[~data[feats[1146]].isnull()]
y = data['TARGET']
del data['TARGET']
print('Shapes : ', data.shape, test.shape)
print('Drop sigle NaN')
for c in data.columns.tolist():
    if data[c].isnull().sum()==1:
        data=data[~data[c].isnull()]
print('Shapes : ', data.shape, test.shape)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # 0 !!!!!
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

#print(data[feats[1134]].isnull().sum())
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]

#data.drop(feats[1134],axis=1,inplace=True)
#test.drop(feats[1134],axis=1,inplace=True)
CATEGORICAL_COLUMNS=[c for c in CATEGORICAL_COLUMNS if c in data.columns.tolist()]
cl=[feats.index(c) for c in CATEGORICAL_COLUMNS]
CATEGORICAL_COLUMNS=cl

roc=pd.DataFrame(columns=['auc','fold'])
print("Training model")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data,y)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    model = CatBoostClassifier(random_seed=42,iterations=5000,metric_period=100,
                               bagging_temperature=0.5,rsm=0.1,one_hot_max_size=15,
                               depth=4, l2_leaf_reg=5,eval_metric='AUC',od_type="Iter",od_wait=600)
    print('Fold '+str(n_fold+1))
    print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
#    if n_fold==3:
    model.fit(trn_x, trn_y,eval_set=(val_x, val_y),
      use_best_model=True,  verbose=200,cat_features=CATEGORICAL_COLUMNS)
    val_pred=model.predict_proba(val_x)
    oof_preds[val_idx] = model.predict_proba(val_x)[:, 1]
    test_pred=model.predict_proba(test[feats])
    sub_preds += test_pred[:, 1] / folds.n_splits
    print("roc_auc_score = {}".format(roc_auc_score(val_y, val_pred[:, 1])))
    roc=roc.append({'auc':roc_auc_score(val_y, oof_preds[val_idx]),'fold':(n_fold+1)},ignore_index=True)
    del model, trn_x, trn_y, val_x, val_y,val_pred,test_pred
    gc.collect()
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))
test['TARGET'] = sub_preds
roc=roc.append({'auc':roc_auc_score(y, oof_preds),'fold':0},ignore_index=True)
roc.to_csv("../output/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time()+3600*7))+"_roc.csv", index=False)
from time import gmtime, strftime
test[['SK_ID_CURR', 'TARGET']].to_csv("../output/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time()+3600*7))+"_cat_submission.csv", index=False)