#Based on  awesome script (https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm) and parameters (https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt)

import pandas as pd
import numpy as np

from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
import gc
import sys
import warnings
#nrows=10000
nrows=None
warnings.filterwarnings('ignore')
gc.enable()
####
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

def reduce_mem(df):
    start_mem = df.memory_usage().sum() / 1024**2
    #print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        #else:
            #df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
   # print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Before: {:.2f} MB, after: {:.2f} MB, decreased by {:.1f}%'.format(start_mem,end_mem,100 * (start_mem - end_mem) / start_mem))
    
    return df
print("Loading data...\n")

feature_drop=pd.read_csv('./feature_importance_gain.csv')
feature_drop=feature_drop[['feature','importance']].groupby('feature').mean().reset_index()
feature_drop=pd.DataFrame(feature_drop.loc[feature_drop['importance']<100,'feature'])
#print(feature_drop['feature'].tolist())
print('Read train and test')
data = pd.read_csv('../output/data_eng.csv',nrows=nrows)
test = pd.read_csv('../output/test_eng.csv',nrows=nrows)


print('Shapes : ', data.shape, test.shape)
#test.info(verbose=True)
#sys.exit(0)
#data.to_csv("data_eng.csv",index=False)
#test.to_csv("test_eng.csv",index=False)
print("Preprocessing...\n")
print('Drop feature')
#dr_feat=[c for c in USELESS_COLUMNS if c in data.columns.tolist()]
#data.drop(dr_feat,axis=1,inplace=True)
#test.drop(dr_feat,axis=1,inplace=True)
#dr_feat=[c for c in HIGHLY_CORRELATED_NUMERICAL_COLUMNS if c in data.columns.tolist()]
#data.drop(dr_feat,axis=1,inplace=True)
#test.drop(dr_feat,axis=1,inplace=True)
dr_feat=[c for c in feature_drop['feature'].tolist() if c in data.columns.tolist()]
data.drop(dr_feat,axis=1,inplace=True)
test.drop(dr_feat,axis=1,inplace=True)
#dr_feat=[c for c in data.columns.tolist() if c.find('bureau_balance')!=-1]
#data.drop(dr_feat,axis=1,inplace=True)
#test.drop(dr_feat,axis=1,inplace=True)

CATEGORICAL_COLUMNS=[c for c in CATEGORICAL_COLUMNS if c in data.columns.tolist()]
print('Shapes : ', data.shape, test.shape)
cat_vars = [(col, data[col].unique().shape[0]) for col in CATEGORICAL_COLUMNS]
print(cat_vars)
#categorical_feats=[c for c in categorical_feats if c in data.columns.tolist()]

y = data['TARGET']
del data['TARGET']
from embedder import preprocessing as pr
from embedder.classification import Embedder
for f_ in CATEGORICAL_COLUMNS:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
#for c in CATEGORICAL_COLUMNS:
#    m=data[c].max()+1
#    data.loc[data[c]==-1,c]=m
#cat_sz = pr.categorize(data)
#emb_sz = pr.size_embeddings(cat_vars)
#data_encoded, encoders = pr.encode_categorical(data)
embedding_dict =pr.pick_emb_dim(cat_vars, max_dim=60)
#data_encoded, encoders = pr.encode_categorical(data)
#data_encoded[CATEGORICAL_COLUMNS].to_csv("dtct1.csv")
#print(embedding_dict)
embedder = Embedder(embedding_dict)
#embedder.fit(data_encoded, y)
sizes = [(var, sz[1]) for var, sz in embedding_dict.items()]
#print(sizes)
#result = [x for sublist in nested_list for x in sublist if x % 3 == 0]
#data_encoded=data[CATEGORICAL_COLUMNS]
feats=data.columns.tolist()
embedder.fit(data[CATEGORICAL_COLUMNS],y
              ,epochs=10,batch_size=10000)
print("transofrm data")
emb_data=embedder.transform(data[CATEGORICAL_COLUMNS]
              ,as_df=True)
print("transofrm test")
emb_test=embedder.transform(test[CATEGORICAL_COLUMNS]
              ,as_df=True)
nc=[c for c in emb_data.columns.tolist() if c not in feats]
emb_data[nc].to_csv("dtct.csv")
emb_test[nc].to_csv("dtct.csv")
data.drop(CATEGORICAL_COLUMNS,axis=1,inplace=True)
test.drop(CATEGORICAL_COLUMNS,axis=1,inplace=True)
data=pd.concat([data,emb_data],axis=1)
test=pd.concat([test,emb_test],axis=1)
data.info(verbose=True)
#emb_data.info(verbose=True)
sys.exit(0)
gc.collect()
    
from lightgbm import LGBMClassifier


#folds = KFold(n_splits=5, shuffle=True, random_state=90210)
folds = StratifiedKFold(n_splits=20, shuffle=True, random_state=90210) # 0 !!!!!
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()
roc=pd.DataFrame(columns=['auc','fold'])
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
from scipy.stats import rankdata
import lightgbm as lgb
print("Training model...\n")
#for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data)):
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data,y)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    
    clf = LGBMClassifier(
        #boosting_type='dart',
        verbose=-1,
        boosting_type='gbdt',
        colsample_bytree=0.05,
        early_stopping_rounds=600,
        is_unbalance=False,
        learning_rate=0.02,
        max_bin=300,
        max_depth=-1,
        metric='auc',
        min_child_samples=70,
        min_gain_to_split=0.5,
        num_leaves=30,
        n_estimators=10000,
        objective='binary',
        reg_alpha=0.0,
        reg_lambda=100.0,
        scale_pos_weight=1.0,
        subsample=1.0,
        subsample_freq=1,
        random_state=n_fold
    )
    
    clf.fit(trn_x, trn_y, callbacks=[lgb.reset_parameter(learning_rate=[(200/(8000+x)) for x in range(10000)])],
            eval_set= [(trn_x, trn_y), (val_x, val_y)],
            eval_metric='auc', verbose=100, early_stopping_rounds=300,categorical_feature=CATEGORICAL_COLUMNS #30
           )
    
    oof_preds[val_idx] = clf.predict_proba(val_x, num_iteration=clf.best_iteration_)[:, 1]
    sub_preds += rankdata(clf.predict_proba(test[feats], num_iteration=clf.best_iteration_)[:, 1]) / folds.n_splits/len(sub_preds)
    
    fold_importance_df = pd.DataFrame()
    fold_importance_df["feature"] = feats
    fold_importance_df["importance"] = clf.feature_importances_
    fold_importance_df["fold"] = n_fold + 1
    feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)
    
    print('Fold %2d AUC : %.6f' % (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
    roc=roc.append({'auc':roc_auc_score(val_y, oof_preds[val_idx]),'fold':(n_fold+1)},ignore_index=True)
    del clf, trn_x, trn_y, val_x, val_y
    gc.collect()
    
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
print('LightGBM full AUC score %.6f' % roc_auc_score(y, oof_preds)) 
roc=roc.append({'auc':roc_auc_score(y, oof_preds),'fold':0},ignore_index=True)

test['TARGET'] = sub_preds
from time import gmtime, strftime
test[['SK_ID_CURR', 'TARGET']].to_csv("../output/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime())+"_first_submission.csv", index=False)

roc.to_csv('../output/'+strftime("%Y-%m-%d_%H-%M-%S", gmtime())+'fold_auc.csv', index=False)