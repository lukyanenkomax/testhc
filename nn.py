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
#import catboost
feats = [f for f in data.columns if f not in ['SK_ID_CURR','TARGET']]
#print(CATEGORICAL_COLUMNS)
#`import catboost
#catboost.__version__
#feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
#data.drop(feats[1134],axis=1,inplace=True)
#test.drop(feats[1134],axis=1,inplace=True)
#data=data[~data[feats[1134]].isnull()]
#data=data[~data[feats[1146]].isnull()]
#y = data['TARGET']
#del data['TARGET']
print('Shapes : ', data.shape, test.shape)
#print('Drop sigle NaN')
#for c in data.columns.tolist():
#    if data[c].isnull().values.sum()<3&data[c].isnull().values.sum()>0:
#        data=data[~data[c].isnull()]
#print('Shapes : ', data.shape, test.shape)
#data=data[~data[feats[1134]].isnull()]
#data=data[~data[feats[1146]].isnull()]
print('Shapes : ', data.shape, test.shape)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=0) # 0 !!!!!
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

#print(data[feats[1134]].isnull().sum())
feature_drop=pd.read_csv('./feature_importance_gain.csv')
feature_drop=feature_drop[['feature','importance']].groupby('feature').mean().reset_index()
feature_drop=pd.DataFrame(feature_drop.loc[feature_drop['importance']<100,'feature'])
feature_drop = [f for f in feature_drop['feature'] if f in data.columns.tolist()]
data=data.drop(feature_drop,axis=1)
test=test.drop(feature_drop,axis=1)
#data.drop(feats[1134],axis=1,inplace=True)
#test.drop(feats[1134],axis=1,inplace=True)
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout, BatchNormalization
from keras.layers.advanced_activations import PReLU
from keras.optimizers import Adam
#print(data.shape)
#print(test.shape)
#data.info(verbose=True)
#test.info(verbose=True)
df=data.append(test)
print(df.shape)
del data,test
gc.collect()
y = df['TARGET']
feats = [f for f in df.columns if f not in ['TARGET','SK_ID_CURR','SK_ID_BUREAU','SK_ID_PREV','index']]
X = df[feats]
print("X shape: ", X.shape, "    y shape:", y.shape)

print("\nPreparing data...")
X = X.fillna(X.mean()).clip(-1e11,1e11)
scaler = MinMaxScaler()
scaler.fit(X)
training = y.notnull()
testing = y.isnull()
X_train = scaler.transform(X[training])
X_test = scaler.transform(X[testing])
y_train = np.array(y[training])
print( X_train.shape, X_test.shape, y_train.shape )


print( 'Setting up neural network...' )
nn = Sequential()
nn.add(Dense(units = 400 , kernel_initializer = 'normal', input_dim = len(feats)))
nn.add(PReLU())
nn.add(Dropout(.3))
nn.add(Dense(units = 160 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units = 64 , kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units = 26, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(units = 12, kernel_initializer = 'normal'))
nn.add(PReLU())
nn.add(BatchNormalization())
nn.add(Dropout(.3))
nn.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))
nn.compile(loss='binary_crossentropy', optimizer='adam')

print( 'Fitting neural network...' )
nn.fit(X_train, y_train, validation_split=0.1, epochs=10, verbose=2)

print( 'Predicting...' )
y_pred = nn.predict(X_test).flatten().clip(0,1)


sys.exit(0)
roc=pd.DataFrame(columns=['auc','fold'])
print("Training model")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data,y)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    model = CatBoostClassifier(random_seed=42,iterations=20000,metric_period=100,
                               bagging_temperature=0.5,rsm=0.1,one_hot_max_size=15,
                               depth=4, l2_leaf_reg=2,eval_metric='AUC',od_type="Iter",od_wait=1200)
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