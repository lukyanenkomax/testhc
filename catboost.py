from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
import gc
import sys
import warnings
data = pd.read_csv('../output/data_eng.csv')
test = pd.read_csv('../output/test_eng.csv')
folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=0) # 0 !!!!!
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])
import catboost
y = data['TARGET']
del data['TARGET']
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
from catboost import CatBoostClassifier
for n_fold, (trn_idx, val_idx) in enumerate(folds.split(data,y)):
    trn_x, trn_y = data[feats].iloc[trn_idx], y.iloc[trn_idx]
    val_x, val_y = data[feats].iloc[val_idx], y.iloc[val_idx]
    model = CatBoostClassifier(random_seed=42,iterations=5000,metric_period=100,
                               bagging_temperature=0.5,
                               depth=4, l2_leaf_reg=5, learning_rate=0.07)
    print('Fold '+str(n_fold+1))
    model.fit(trn_x, trn_y,eval_set=(val_x, val_y),use_best_model=True,  verbose=200)
    val_pred=model.predict_proba(val_x)
    oof_preds[val_idx] = model.predict_proba(val_x)[:, 1]
    test_pred=model.predict_proba(test[feats])
    sub_preds += test_pred[:, 1] / folds.n_splits
    print("roc_auc_score = {}".format(roc_auc_score(val_y, val_pred[:, 1])))
    del model, trn_x, trn_y, val_x, val_y,val_pred,test_pred
    gc.collect()
print('Full AUC score %.6f' % roc_auc_score(y, oof_preds))