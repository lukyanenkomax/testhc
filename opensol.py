#Based on  awesome script (https://www.kaggle.com/ogrellier/good-fun-with-ligthgbm) and parameters (https://www.kaggle.com/tilii7/olivier-lightgbm-parameters-by-bayesian-opt)

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder 
from functools import partial
import gc
import os
import subprocess
import sys
from time import gmtime, strftime,time
gc.enable()

import warnings
warnings.filterwarnings("ignore")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
#debug
#nrows=5000
nrows=None
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
def LabelEncoding_Cat(df):
    lb=LabelEncoder()
    df=df.copy()
    Cat_Var=df.select_dtypes('object').columns.tolist()
    for col in Cat_Var:
        df[col]=lb.fit_transform(df[col].astype('str'))
    return df    

def Fill_NA(df):
    df=df.copy()
    Num_Features=df.select_dtypes(['float64','int64']).columns.tolist()
    df[Num_Features]= df[Num_Features].fillna(-999)
    return df
    
def application_train_test(df):
    return df
    
def one_hot_encoder(df, nan_as_category = True):
    original_columns = list(df.columns)
    categorical_columns = [col for col in df.columns if df[col].dtype == 'object']
    df = pd.get_dummies(df, columns= categorical_columns, dummy_na= nan_as_category)
    new_columns = [c for c in df.columns if c not in original_columns]
    return df, new_columns
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
import multiprocessing as mp
from functools import reduce
def parallel_apply(groups, func, index_name='Index', num_workers=1, chunk_size=100000):
    n_chunks = np.ceil(1.0 * groups.ngroups / chunk_size)
    indeces, features = [], []
    for index_chunk, groups_chunk in tqdm(chunk_groups(groups, chunk_size), total=n_chunks):
        with mp.pool.Pool(num_workers) as executor:
            features_chunk = executor.map(func, groups_chunk)
        features.extend(features_chunk)
        indeces.extend(index_chunk)

    features = pd.DataFrame(features)
    features.index = indeces
    features.index.name = index_name
    return features
def chunk_groups(groupby_object, chunk_size):
    n_groups = groupby_object.ngroups
    group_chunk, index_chunk = [], []
    for i, (index, df) in enumerate(groupby_object):
        group_chunk.append(df)
        index_chunk.append(index)

        if (i + 1) % chunk_size == 0 or i + 1 == n_groups:
            group_chunk_, index_chunk_ = group_chunk.copy(), index_chunk.copy()
            group_chunk, index_chunk = [], []
            yield index_chunk_, group_chunk_
def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()

    return features

print('Loading train and test')
data = reduce_mem(pd.read_csv('../input/application_train.csv')
                 .sort_values('SK_ID_CURR').reset_index(drop = True).loc[:nrows, :])
test = reduce_mem(pd.read_csv('../input/application_test.csv')
                 .sort_values('SK_ID_CURR').reset_index(drop = True).loc[:nrows, :])
data['INCOMPLETE_APP']=data.isnull().sum(axis=1)
test['INCOMPLETE_APP']=test.isnull().sum(axis=1)
#########
feature_drop=pd.read_csv('./feature_importance_rank.csv')
#eature_drop=pd.read_csv('./f_importance.csv')
#eature_drop=feature_drop['feature','importance'].groupby('feature').mean().reset_index()
feature_drop=pd.DataFrame(feature_drop.loc[feature_drop['importance']<5,'feature'])
print("Loading bureau...\n")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
bureau= pd.read_csv("../input/bureau.csv").sort_values(['SK_ID_CURR', 'SK_ID_BUREAU']).reset_index(drop = True).loc[:nrows, :]
print("Preprocessing bureau...\n")
bureau['bureau_credit_active_binary'] = (bureau['CREDIT_ACTIVE'] != 'Closed').astype(int)
bureau['bureau_credit_enddate_binary'] = (bureau['DAYS_CREDIT_ENDDATE'] > 0).astype(int)
bureau['DAYS_CREDIT_ENDDATE'][bureau['DAYS_CREDIT_ENDDATE'] < -40000] = np.nan
bureau['DAYS_CREDIT_UPDATE'][bureau['DAYS_CREDIT_UPDATE'] < -40000] = np.nan
bureau['DAYS_ENDDATE_FACT'][bureau['DAYS_ENDDATE_FACT'] < -40000] = np.nan
groupby_SK_ID_CURR = bureau.groupby(by=['SK_ID_CURR'])
group_object = groupby_SK_ID_CURR['DAYS_CREDIT'].agg('count').reset_index()
features = pd.DataFrame({'SK_ID_CURR':bureau['SK_ID_CURR'].unique()})
group_object.rename(index=str, columns={'DAYS_CREDIT': 'bureau_number_of_past_loans'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object = groupby_SK_ID_CURR['CREDIT_TYPE'].agg('nunique').reset_index()
group_object.rename(index=str, columns={'CREDIT_TYPE': 'bureau_number_of_loan_types'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
features['bureau_average_of_past_loans_per_type'] = \
    features['bureau_number_of_past_loans'] / features['bureau_number_of_loan_types']
group_object = groupby_SK_ID_CURR['bureau_credit_active_binary'].agg('mean').reset_index()
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_DEBT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_CREDIT_SUM_DEBT': 'bureau_total_customer_debt'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_CREDIT_SUM': 'bureau_total_customer_credit'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
features['bureau_debt_credit_ratio'] = \
    features['bureau_total_customer_debt'] / features['bureau_total_customer_credit']
group_object = groupby_SK_ID_CURR['AMT_CREDIT_SUM_OVERDUE'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_CREDIT_SUM_OVERDUE': 'bureau_total_customer_overdue'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
features['bureau_overdue_debt_ratio'] = \
    features['bureau_total_customer_overdue'] / features['bureau_total_customer_debt']
group_object = groupby_SK_ID_CURR['CNT_CREDIT_PROLONG'].agg('sum').reset_index()
group_object.rename(index=str, columns={'CNT_CREDIT_PROLONG': 'bureau_average_creditdays_prolonged'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object = groupby_SK_ID_CURR['bureau_credit_enddate_binary'].agg('mean').reset_index()
group_object.rename(index=str, columns={'bureau_credit_enddate_binary': 'bureau_credit_enddate_percentage'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
BUREAU_AGGREGATION_RECIPIES = [('CREDIT_TYPE', 'count'),
                               ('CREDIT_ACTIVE', 'size')
                               ]
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_CREDIT_SUM',
                   'AMT_CREDIT_SUM_DEBT',
                   'AMT_CREDIT_SUM_LIMIT',
                   'AMT_CREDIT_SUM_OVERDUE',
                   'AMT_CREDIT_MAX_OVERDUE',
                   'CNT_CREDIT_PROLONG',
                   'CREDIT_DAY_OVERDUE',
                   'DAYS_CREDIT',
                   'DAYS_CREDIT_ENDDATE',
                   'DAYS_CREDIT_UPDATE'
                   ]:
        BUREAU_AGGREGATION_RECIPIES.append((select, agg))
BUREAU_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], BUREAU_AGGREGATION_RECIPIES)]
groupby_aggregate_names = []
#bureau_agg=pd.DataFrame(bureau['SK_ID_CURR'].drop_duplicates().reset_index(drop=True))
for groupby_cols, specs in BUREAU_AGGREGATION_RECIPIES:
    group_object = bureau.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        features = features.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
features=reduce_mem(features)
#del bureau
gc.collect()
data = data.merge(right=features, how='left', on='SK_ID_CURR')
test = test.merge(right=features, how='left', on='SK_ID_CURR')
print('Shapes : ', data.shape, test.shape)
#######
print("Loading card...\n")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
credit_card=pd.read_csv("../input/credit_card_balance.csv").sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:nrows, :]
credit_card['AMT_DRAWINGS_ATM_CURRENT'][credit_card['AMT_DRAWINGS_ATM_CURRENT'] < 0] = np.nan
credit_card['AMT_DRAWINGS_CURRENT'][credit_card['AMT_DRAWINGS_CURRENT'] < 0] = np.nan
credit_card['number_of_instalments'] = credit_card.groupby(
    by=['SK_ID_CURR', 'SK_ID_PREV'])['CNT_INSTALMENT_MATURE_CUM'].agg('max').reset_index()[
    'CNT_INSTALMENT_MATURE_CUM']
credit_card['credit_card_max_loading_of_credit_limit'] = credit_card.groupby(
    by=['SK_ID_CURR', 'SK_ID_PREV', 'AMT_CREDIT_LIMIT_ACTUAL']).apply(
    lambda x: x.AMT_BALANCE.max() / x.AMT_CREDIT_LIMIT_ACTUAL.max()).reset_index()[0]
features = pd.DataFrame({'SK_ID_CURR':credit_card['SK_ID_CURR'].unique()})
group_object = credit_card.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].agg('nunique').reset_index()
group_object.rename(index=str, columns={'SK_ID_PREV': 'credit_card_number_of_loans'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object= credit_card.groupby(by=['SK_ID_CURR'])['number_of_instalments'].sum().reset_index()
group_object.rename(index=str, columns={'number_of_instalments': 'credit_card_total_instalments'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
features['credit_card_installments_per_loan'] = (
    features['credit_card_total_instalments'] / features['credit_card_number_of_loans'])
group_object = credit_card.groupby(by=['SK_ID_CURR'])['credit_card_max_loading_of_credit_limit'].agg('mean').reset_index()
group_object.rename(index=str, columns={'credit_card_max_loading_of_credit_limit': 'credit_card_avg_loading_of_credit_limit'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object = credit_card.groupby(
    by=['SK_ID_CURR'])['SK_DPD'].agg('mean').reset_index()
group_object.rename(index=str, columns={'SK_DPD': 'credit_card_average_of_days_past_due'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_ATM_CURRENT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_DRAWINGS_ATM_CURRENT': 'credit_card_drawings_atm'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

group_object = credit_card.groupby(by=['SK_ID_CURR'])['AMT_DRAWINGS_CURRENT'].agg('sum').reset_index()
group_object.rename(index=str, columns={'AMT_DRAWINGS_CURRENT': 'credit_card_drawings_total'},inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
features['credit_card_cash_card_ratio'] = features['credit_card_drawings_atm'] / features['credit_card_drawings_total']
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_BALANCE',
                   'AMT_CREDIT_LIMIT_ACTUAL',
                   'AMT_DRAWINGS_ATM_CURRENT',
                   'AMT_DRAWINGS_CURRENT',
                   'AMT_DRAWINGS_OTHER_CURRENT',
                   'AMT_DRAWINGS_POS_CURRENT',
                   'AMT_PAYMENT_CURRENT',
                   'CNT_DRAWINGS_ATM_CURRENT',
                   'CNT_DRAWINGS_CURRENT',
                   'CNT_DRAWINGS_OTHER_CURRENT',
                   'CNT_INSTALMENT_MATURE_CUM',
                   'MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES)]
groupby_aggregate_names = []
for groupby_cols, specs in CREDIT_CARD_BALANCE_AGGREGATION_RECIPIES:
    group_object = credit_card.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        features = features.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
#features.info(verbose=True)
features=reduce_mem(features)
data = data.merge(right=features, how='left', on='SK_ID_CURR')
test = test.merge(right=features, how='left', on='SK_ID_CURR')
del credit_card
gc.collect()
print('Shapes : ', data.shape, test.shape)
#######
print("Loading pos...\n")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
pos_cash_balance=pd.read_csv("../input/POS_CASH_balance.csv").sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:nrows, :]

POS_CASH_BALANCE_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['MONTHS_BALANCE',
                   'SK_DPD',
                   'SK_DPD_DEF'
                   ]:
        POS_CASH_BALANCE_AGGREGATION_RECIPIES.append((select, agg))
POS_CASH_BALANCE_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], POS_CASH_BALANCE_AGGREGATION_RECIPIES)]
features = pd.DataFrame({'SK_ID_CURR':pos_cash_balance['SK_ID_CURR'].unique()})
groupby_aggregate_names = []
for groupby_cols, specs in POS_CASH_BALANCE_AGGREGATION_RECIPIES:
    group_object = pos_cash_balance.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        features = features.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
pos_cash_sorted = pos_cash_balance.sort_values(['SK_ID_CURR', 'MONTHS_BALANCE'])
group_object = pos_cash_sorted.groupby('SK_ID_CURR')['CNT_INSTALMENT_FUTURE'].last().reset_index()
group_object.rename(index=str,
                    columns={'CNT_INSTALMENT_FUTURE': 'pos_cash_remaining_installments'},
                    inplace=True)

features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
pos_cash_balance['is_contract_status_completed'] = pos_cash_balance['NAME_CONTRACT_STATUS'] == 'Completed'
group_object = pos_cash_balance.groupby(['SK_ID_CURR'])['is_contract_status_completed'].sum().reset_index()
group_object.rename(index=str,
                    columns={'is_contract_status_completed': 'pos_cash_completed_contracts'},
                    inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
pos_cash_balance['pos_cash_paid_late'] = (pos_cash_balance['SK_DPD'] > 0).astype(int)
pos_cash_balance['pos_cash_paid_late_with_tolerance'] = (pos_cash_balance['SK_DPD_DEF'] > 0).astype(int)
groupby = pos_cash_balance.groupby(['SK_ID_CURR'])
def last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'all_installment_'
            gr_period = gr_.copy()
        else:
            period_name = 'last_{}_'.format(period)
            gr_period = gr_.iloc[:period]

        features = add_features_in_group(features, gr_period, 'pos_cash_paid_late',
                                             ['count', 'mean'],
                                             period_name)
        features = add_features_in_group(features, gr_period, 'pos_cash_paid_late_with_tolerance',
                                             ['count', 'mean'],
                                             period_name)
        features = add_features_in_group(features, gr_period, 'SK_DPD',
                                             ['sum', 'mean', 'max', 'min', 'median'],
                                             period_name)
        features = add_features_in_group(features, gr_period, 'SK_DPD_DEF',
                                             ['sum', 'mean', 'max', 'min','median'],
                                             period_name)
    return features

func = partial(last_k_installment_features, periods=[1, 6, 12, 10e16])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')
def last_loan_features(gr):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)
    last_installment_id = gr_['SK_ID_PREV'].iloc[0]
    gr_ = gr_[gr_['SK_ID_PREV'] == last_installment_id]

    features={}
    features = add_features_in_group(features, gr_, 'pos_cash_paid_late',
                                         ['count', 'sum', 'mean'],
                                         'last_loan_')
    features = add_features_in_group(features, gr_, 'pos_cash_paid_late_with_tolerance',
                                         ['sum', 'mean'],
                                         'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
    features = add_features_in_group(features, gr_, 'SK_DPD_DEF',
                                         ['sum', 'mean', 'max', 'min', 'std'],
                                         'last_loan_')
    return features

g = parallel_apply(groupby, last_loan_features, index_name='SK_ID_CURR', num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')
def trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]

        features = add_trend_feature(features, gr_period,
                                         'SK_DPD', '{}_period_trend_'.format(period)
                                         )
        features = add_trend_feature(features, gr_period,
                                         'SK_DPD_DEF', '{}_period_trend_'.format(period)
                                         )
    return features

def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features

func = partial(trend_in_last_k_installment_features, periods=[1,6,12,30,60])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')
features=reduce_mem(features)
data = data.merge(right=features, how='left', on='SK_ID_CURR')
test = test.merge(right=features, how='left', on='SK_ID_CURR')
del pos_cash_balance
print('Shapes : ', data.shape, test.shape)
gc.collect()
print("Loading installments...\n")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
from scipy.stats import skew, kurtosis, iqr
installments_ = pd.read_csv('../input/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:nrows, :]
#installments_ = pd.read_csv('../input/installments_payments.csv').sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:1000000, :]
installments_['instalment_paid_late_in_days'] = installments_['DAYS_ENTRY_PAYMENT'] - installments_['DAYS_INSTALMENT'] 
installments_['instalment_paid_late'] = (installments_['instalment_paid_late_in_days'] > 0).astype(int)
installments_['instalment_paid_over_amount'] = installments_['AMT_PAYMENT'] - installments_['AMT_INSTALMENT']
installments_['instalment_paid_over'] = (installments_['instalment_paid_over_amount'] > 0).astype(int)

def add_features(feature_name, aggs, features, feature_names, groupby):
    feature_names.extend(['{}_{}'.format(feature_name, agg) for agg in aggs])
    for agg in aggs:
        if agg == 'kurt':
            agg_func = kurtosis
        elif agg == 'iqr':
            agg_func = iqr
        else:
            agg_func = agg
        g = groupby[feature_name].agg(agg_func).reset_index().rename(index=str,
                                                                columns={feature_name: '{}_{}'.format(feature_name,
                                                                                                      agg)})
        features = features.merge(g, on='SK_ID_CURR', how='left')
    return features, feature_names


def add_features_in_group(features, gr_, feature_name, aggs, prefix):
    for agg in aggs:
        if agg == 'sum':
            features['{}{}_sum'.format(prefix, feature_name)] = gr_[feature_name].sum()
        elif agg == 'mean':
            features['{}{}_mean'.format(prefix, feature_name)] = gr_[feature_name].mean()
        elif agg == 'max':
            features['{}{}_max'.format(prefix, feature_name)] = gr_[feature_name].max()
        elif agg == 'min':
            features['{}{}_min'.format(prefix, feature_name)] = gr_[feature_name].min()
        elif agg == 'std':
            features['{}{}_std'.format(prefix, feature_name)] = gr_[feature_name].std()
        elif agg == 'count':
            features['{}{}_count'.format(prefix, feature_name)] = gr_[feature_name].count()
        elif agg == 'skew':
            features['{}{}_skew'.format(prefix, feature_name)] = skew(gr_[feature_name])
        elif agg == 'kurt':
            features['{}{}_kurt'.format(prefix, feature_name)] = kurtosis(gr_[feature_name])
        elif agg == 'iqr':
            features['{}{}_iqr'.format(prefix, feature_name)] = iqr(gr_[feature_name])
        elif agg == 'median':
            features['{}{}_median'.format(prefix, feature_name)] = gr_[feature_name].median()
    return features

features = pd.DataFrame({'SK_ID_CURR':installments_['SK_ID_CURR'].unique()})
groupby = installments_.groupby(['SK_ID_CURR'])
feature_names = []

features, feature_names = add_features('NUM_INSTALMENT_VERSION', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_late_in_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_late', ['sum','mean'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_over_amount', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                     features, feature_names, groupby)

features, feature_names = add_features('instalment_paid_over', ['sum','mean'],
                                     features, feature_names, groupby)
                                     
def last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]
        features = add_features_in_group(features,gr_period, 'NUM_INSTALMENT_VERSION', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        
        features = add_features_in_group(features,gr_period, 'instalment_paid_late_in_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_late', 
                                     ['count','mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_over_amount', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period,'instalment_paid_over', 
                                     ['count','mean'],
                                         'last_{}_'.format(period))      
    
          
    
    return features
print('Stage 1')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
func = partial(last_k_instalment_features, periods=[1,5,10,20,50,100])

g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')
from sklearn.linear_model import LinearRegression

def trend_in_last_k_instalment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    
    features = {}

    for period in periods:
        gr_period = gr_.iloc[:period]


        features = _add_trend_feature(features,gr_period,
                                      'instalment_paid_late_in_days','{}_period_trend_'.format(period)
                                     )
        features = _add_trend_feature(features,gr_period,
                                      'instalment_paid_over_amount','{}_period_trend_'.format(period)
                                     )
    return features

def _add_trend_feature(features,gr,feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0,len(y)).reshape(-1,1)
        lr = LinearRegression()
        lr.fit(x,y)
        trend = lr.coef_[0]
    except:
        trend=np.nan
    features['{}{}'.format(prefix,feature_name)] = trend
    return features
print('Stage 2')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
func = partial(trend_in_last_k_instalment_features, periods=[10,50,100,500])

g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

def last_k_instalment_features_with_fractions(gr, periods, fraction_periods):
    gr_ = gr.copy()
    gr_.sort_values(['DAYS_INSTALMENT'],ascending=False, inplace=True)
    
    features = {}
    for period in periods:
        gr_period = gr_.iloc[:period]
        features = add_features_in_group(features,gr_period, 'NUM_INSTALMENT_VERSION', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        
        features = add_features_in_group(features,gr_period, 'instalment_paid_late_in_days', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_late', 
                                     ['count','mean'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period ,'instalment_paid_over_amount', 
                                       ['sum','mean','max','min','std', 'median','skew', 'kurt','iqr'],
                                         'last_{}_'.format(period))
        features = add_features_in_group(features,gr_period,'instalment_paid_over', 
                                     ['count','mean'],
                                         'last_{}_'.format(period))             
    
    for short_period, long_period in fraction_periods:
        short_feature_names = _get_feature_names(features, short_period)
        long_feature_names = _get_feature_names(features, long_period)
        
        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk ='_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = safe_div(features[short_feature], features[long_feature])
    return pd.Series(features)

def _get_feature_names(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])


def safe_div(a,b):
    try:
        return float(a)/float(b)
    except:
        return 0.0
print('Stage 3')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
func = partial(last_k_instalment_features_with_fractions, 
               periods=[1,5,10,20,50,100],
               fraction_periods=[(5,20),(5,50),(10,100)])

g = parallel_apply(groupby, func, index_name='SK_ID_CURR',
                   num_workers=12, chunk_size=1000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')
     
from sys import exit as sys_exit
features=reduce_mem(features)
data = data.merge(right=features, how='left', on='SK_ID_CURR')
test = test.merge(right=features, how='left', on='SK_ID_CURR')
del installments_,groupby
gc.collect()
#features.info(verbose=True)
print('Shapes : ', data.shape, test.shape)
print('Loading previous application')
previous_application =pd.read_csv("../input/previous_application.csv").sort_values(['SK_ID_CURR', 'SK_ID_PREV']).reset_index(drop = True).loc[:nrows, :]
previous_application['DAYS_FIRST_DRAWING'].replace(365243, np.nan, inplace=True)
previous_application['DAYS_FIRST_DUE'].replace(365243, np.nan, inplace=True)
previous_application['DAYS_LAST_DUE_1ST_VERSION'].replace(365243, np.nan, inplace=True)
previous_application['DAYS_LAST_DUE'].replace(365243, np.nan, inplace=True)
previous_application['DAYS_TERMINATION'].replace(365243, np.nan, inplace=True)               
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = []
for agg in ['mean', 'min', 'max', 'sum', 'var']:
    for select in ['AMT_ANNUITY',
                   'AMT_APPLICATION',
                   'AMT_CREDIT',
                   'AMT_DOWN_PAYMENT',
                   'AMT_GOODS_PRICE',
                   'CNT_PAYMENT',
                   'DAYS_DECISION',
                   'HOUR_APPR_PROCESS_START',
                   'RATE_DOWN_PAYMENT'
                   ]:
        PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, agg))
prev_status=previous_application['NAME_CONTRACT_STATUS']
previous_application,new_prev_col=one_hot_encoder(previous_application)
previous_application['NAME_CONTRACT_STATUS']=prev_status
for select in new_prev_col:
    PREVIOUS_APPLICATION_AGGREGATION_RECIPIES.append((select, 'sum'))
PREVIOUS_APPLICATION_AGGREGATION_RECIPIES = [(['SK_ID_CURR'], PREVIOUS_APPLICATION_AGGREGATION_RECIPIES)]
features = pd.DataFrame({'SK_ID_CURR': previous_application['SK_ID_CURR'].unique()})
groupby_aggregate_names = []
for groupby_cols, specs in PREVIOUS_APPLICATION_AGGREGATION_RECIPIES:
    group_object = previous_application.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        features = features.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
#features.info(verbose=True)
numbers_of_applications = [1, 3, 5]
prev_applications_sorted = previous_application.sort_values(['SK_ID_CURR', 'DAYS_DECISION'])
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])['SK_ID_PREV'].nunique().reset_index()
group_object.rename(index=str,
                    columns={'SK_ID_PREV': 'previous_application_number_of_prev_application'},
                    inplace=True)
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
prev_applications_sorted['previous_application_prev_was_approved'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Approved').astype('int')
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    'previous_application_prev_was_approved'].last().reset_index()
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
prev_applications_sorted['previous_application_prev_was_refused'] = (
        prev_applications_sorted['NAME_CONTRACT_STATUS'] == 'Refused').astype('int')
group_object = prev_applications_sorted.groupby(by=['SK_ID_CURR'])[
    'previous_application_prev_was_refused'].last().reset_index()
features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
for number in numbers_of_applications:
    prev_applications_tail = prev_applications_sorted.groupby(by=['SK_ID_CURR']).tail(number)

    group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['CNT_PAYMENT'].mean().reset_index()
    group_object.rename(index=str, columns={
        'CNT_PAYMENT': 'previous_application_term_of_last_{}_credits_mean'.format(number)},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_DECISION'].mean().reset_index()
    group_object.rename(index=str, columns={
        'DAYS_DECISION': 'previous_application_days_decision_about_last_{}_credits_mean'.format(number)},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')

    group_object = prev_applications_tail.groupby(by=['SK_ID_CURR'])['DAYS_FIRST_DRAWING'].mean().reset_index()
    group_object.rename(index=str, columns={
        'DAYS_FIRST_DRAWING': 'previous_application_days_first_drawing_last_{}_credits_mean'.format(number)},
                        inplace=True)
    features = features.merge(group_object, on=['SK_ID_CURR'], how='left')
del  previous_application,prev_applications_sorted
gc.collect()
features=reduce_mem(features)
features.info(verbose=True)
data = data.merge(right=features, how='left', on='SK_ID_CURR')
test = test.merge(right=features, how='left', on='SK_ID_CURR')
print('Shapes : ', data.shape, test.shape)
print('Loading bureau balance...')
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
bureau_balance = pd.read_csv("../input/bureau_balance.csv").sort_values('SK_ID_BUREAU').reset_index(drop = True).loc[:nrows, :]
bureau_balance = bureau_balance.merge(bureau[['SK_ID_CURR', 'SK_ID_BUREAU']], on='SK_ID_BUREAU', how='right')
del bureau
gc.collect()

def _status_to_int(status):
    if status in ['X', 'C']:
        return 0
    if pd.isnull(status):
        return np.nan
    return int(status)

bureau_balance['bureau_balance_dpd_level'] = bureau_balance['STATUS'].apply(_status_to_int)
bureau_balance['bureau_balance_status_unknown'] = (bureau_balance['STATUS'] == 'X').astype(int)
bureau_balance['bureau_balance_no_history'] = bureau_balance['MONTHS_BALANCE'].isnull().astype(int)

groupby = bureau_balance.groupby(['SK_ID_CURR'])

features = pd.DataFrame({'SK_ID_CURR': bureau_balance['SK_ID_CURR'].unique()})
g = groupby['bureau_balance_no_history'].all().astype(int).reset_index()
g.rename(index=str, columns={'bureau_balance_no_history': 'bureau_balance_no_history'}, inplace=True)
features = features.merge(g, on=['SK_ID_CURR'], how='left')

g = groupby['bureau_balance_no_history'].any().astype(int).reset_index()
g.rename(index=str, columns={'bureau_balance_no_history': 'bureau_balance_partial_history'}, inplace=True)
features = features.merge(g, on=['SK_ID_CURR'], how='left')

def last_k_installment_features(gr, periods):
    gr_ = gr.copy()

    features = {}
    for period in periods:
        if period > 10e10:
            period_name = 'all_installment_'
            gr_period = gr_.copy()
        else:
            period_name = 'last_{}_'.format(period)
            gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

        features = add_features_in_group(features, gr_period, 'bureau_balance_dpd_level',
                                             ['sum', 'mean', 'max', 'std', 'skew', 'kurt'],
                                             period_name)
        features = add_features_in_group(features, gr_period, 'bureau_balance_status_unknown',
                                             ['sum', 'mean'],
                                             period_name)
    return features
func = partial(last_k_installment_features, periods=[6, 12, 24, 60, 10e16])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

def trend_in_last_k_installment_features(gr, periods):
    gr_ = gr.copy()
    gr_.sort_values(['MONTHS_BALANCE'], ascending=False, inplace=True)

    features = {}
    for period in periods:
        gr_period = gr_[gr_['MONTHS_BALANCE'] >= (-1) * period]

        features = add_trend_feature(features, gr_period,
                                         'bureau_balance_dpd_level', '{}_period_trend_'.format(period)
                                         )
    return features

def add_trend_feature(features, gr, feature_name, prefix):
    y = gr[feature_name].values
    try:
        x = np.arange(0, len(y)).reshape(-1, 1)
        lr = LinearRegression()
        lr.fit(x, y)
        trend = lr.coef_[0]
    except:
        trend = np.nan
    features['{}{}'.format(prefix, feature_name)] = trend
    return features
func = partial(trend_in_last_k_installment_features, periods=[6,12,24,60])
g = parallel_apply(groupby, func, index_name='SK_ID_CURR', num_workers=12, chunk_size=10000).reset_index()
features = features.merge(g, on='SK_ID_CURR', how='left')

def last_k_instalment_fractions(old_features, fraction_periods):
    features = old_features[['SK_ID_CURR']].copy()
    
    for short_period, long_period in fraction_periods:
        short_feature_names = _get_feature_names(old_features, short_period)
        long_feature_names = _get_feature_names(old_features, long_period)
        
        for short_feature, long_feature in zip(short_feature_names, long_feature_names):
            old_name_chunk = '_{}_'.format(short_period)
            new_name_chunk ='_{}by{}_fraction_'.format(short_period, long_period)
            fraction_feature_name = short_feature.replace(old_name_chunk, new_name_chunk)
            features[fraction_feature_name] = old_features[short_feature]/old_features[long_feature]
    return pd.DataFrame(features).fillna(0.0)

def _get_feature_names(features, period):
    return sorted([feat for feat in features.keys() if '_{}_'.format(period) in feat])
g = last_k_instalment_fractions(features, fraction_periods=[(6, 12), (6, 24), (12,24), (12, 60)])
features = features.merge(g, on='SK_ID_CURR', how='left')
features=reduce_mem(features)
#features.info(verbose=True)
data = data.merge(right=features, how='left', on='SK_ID_CURR')
test = test.merge(right=features, how='left', on='SK_ID_CURR')
print('Shapes : ', data.shape, test.shape)
del features
gc.collect()
def process_app_train(X):
    X['CODE_GENDER'].replace('XNA',np.nan, inplace=True)
    X['DAYS_EMPLOYED'].replace(365243, np.nan, inplace=True)
    X['DAYS_EMPLOYED'] = X['DAYS_EMPLOYED'].map(lambda x: x if x < 25000 else np.nan)
    X['DAYS_LAST_PHONE_CHANGE'].replace(0, np.nan, inplace=True)
    X['NAME_FAMILY_STATUS'].replace('Unknown', np.nan, inplace=True)
    X['ORGANIZATION_TYPE'].replace('XNA', np.nan, inplace=True)
    X['annuity_income_percentage'] = X['AMT_ANNUITY'] / X['AMT_INCOME_TOTAL']
    X['car_to_birth_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_BIRTH']    
    X['car_to_employ_ratio'] = X['OWN_CAR_AGE'] / X['DAYS_EMPLOYED']
    X['children_ratio'] = X['CNT_CHILDREN'] / X['CNT_FAM_MEMBERS']
    X['credit_to_annuity_ratio'] = X['AMT_CREDIT'] / X['AMT_ANNUITY']
    X['credit_to_goods_ratio'] = X['AMT_CREDIT'] / X['AMT_GOODS_PRICE']
    X['credit_to_income_ratio'] = X['AMT_CREDIT'] / X['AMT_INCOME_TOTAL']
    X['days_employed_percentage'] = X['DAYS_EMPLOYED'] / X['DAYS_BIRTH']
    X['income_credit_percentage'] = X['AMT_INCOME_TOTAL'] / X['AMT_CREDIT']
    X['income_per_child'] = X['AMT_INCOME_TOTAL'] / (1 + X['CNT_CHILDREN'])
    X['income_per_person'] = X['AMT_INCOME_TOTAL'] / X['CNT_FAM_MEMBERS']
    X['payment_rate'] = X['AMT_ANNUITY'] / X['AMT_CREDIT']
    X['phone_to_birth_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_BIRTH']
    X['phone_to_employ_ratio'] = X['DAYS_LAST_PHONE_CHANGE'] / X['DAYS_EMPLOYED']
    X['long_employment'] = (X['DAYS_EMPLOYED'] < -2000).astype(int)
    X['cnt_non_child'] = X['CNT_FAM_MEMBERS'] - X['CNT_CHILDREN']
    X['child_to_non_child_ratio'] = X['CNT_CHILDREN'] / X['cnt_non_child']
    X['income_per_non_child'] = X['AMT_INCOME_TOTAL'] / X['cnt_non_child']
    X['credit_per_person'] = X['AMT_CREDIT'] / X['CNT_FAM_MEMBERS']
    X['credit_per_child'] = X['AMT_CREDIT'] / (1 + X['CNT_CHILDREN'])
    X['credit_per_non_child'] = X['AMT_CREDIT'] / X['cnt_non_child']
    X['external_sources_weighted'] = X.EXT_SOURCE_1 * 2 + X.EXT_SOURCE_2 * 3 + X.EXT_SOURCE_3 * 4
    for function_name in ['min', 'max', 'sum', 'mean', 'nanmedian']:
        X['external_sources_{}'.format(function_name)] = eval('np.{}'.format(function_name))(
                X[['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']], axis=1)
    X['credit_pos_cash_inst_ratio'] = X['credit_to_annuity_ratio'] /(1+ X['pos_cash_remaining_installments'])
    X['bureau_debt_income_ratio']=X['bureau_total_customer_debt']/X['AMT_INCOME_TOTAL']
    X['bureau_app_credit_ratio']=X['bureau_total_customer_credit']/X['AMT_CREDIT']
    return X
print("Preprocessing train and test data...\n")
print(strftime("%Y-%m-%d %H:%M:%S", gmtime(time()+3600*7)))
data=process_app_train(data)
test=process_app_train(test)
#categorical feats in train and test
categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
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
start_mem = data.memory_usage().sum() / 1024**2
print('Memory usage of data before agg {:.2f} MB'.format(start_mem))
print('Shapes : ', data.shape, test.shape)
for groupby_cols, specs in AGGREGATION_RECIPIES:
    group_object = data.groupby(groupby_cols)
    for select, agg in specs:
        groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
        data = data.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        test = test.merge(group_object[select]
                              .agg(agg)
                              .reset_index()
                              .rename(index=str,
                                      columns={select: groupby_aggregate_name})
                              [groupby_cols + [groupby_aggregate_name]],
                              on=groupby_cols,
                              how='left')
        groupby_aggregate_names.append(groupby_aggregate_name)
diff_feature_names = []
for groupby_cols, specs in AGGREGATION_RECIPIES:
    for select, agg in specs:
        if agg in ['mean','median','max','min']:
            groupby_aggregate_name = '{}_{}_{}'.format('_'.join(groupby_cols), agg, select)
            diff_name = '{}_diff'.format(groupby_aggregate_name)
            abs_diff_name = '{}_abs_diff'.format(groupby_aggregate_name)
            data[diff_name] = data[select] - data[groupby_aggregate_name] 
            data[abs_diff_name] = np.abs(data[select] - data[groupby_aggregate_name])
            test[diff_name] = test[select] - test[groupby_aggregate_name] 
            test[abs_diff_name] = np.abs(test[select] - test[groupby_aggregate_name]) 
            diff_feature_names.append(diff_name)
            diff_feature_names.append(abs_diff_name)
start_mem = data.memory_usage().sum() / 1024**2
print('Memory usage of data after agg {:.2f} MB'.format(start_mem))
print('Shapes : ', data.shape, test.shape)
#sys_exit(0)


from sklearn.utils import shuffle
data=shuffle(data,random_state=1)
categorical_feats = [
    f for f in data.columns if data[f].dtype == 'object'
]
categorical_feats
for f_ in categorical_feats:
    data[f_], indexer = pd.factorize(data[f_])
    test[f_] = indexer.get_indexer(test[f_])
#print('Merge data')   


print('Drop feature')
#feature_drop=pd.read_csv('../input/feature-importance-min/feature_importance_min.csv')
dr_feat=[c for c in feature_drop['feature'].tolist() if c in data.columns.tolist()]
data.drop(dr_feat,axis=1,inplace=True)
test.drop(dr_feat,axis=1,inplace=True)
print('Shapes : ', data.shape, test.shape)
CATEGORICAL_COLUMNS=[c for c in CATEGORICAL_COLUMNS if c in data.columns.tolist()]
print('Save data')
data.to_csv("../output/data_eng.csv",index=False)
test.to_csv("../output/test_eng.csv",index=False)
y = data['TARGET']
del data['TARGET']
#del(avg_prev,avg_bureau,avg_cred_card_bal,avg_pos_cash_bal,avg_installments_payments)
gc.collect()

from lightgbm import LGBMClassifier


#folds = KFold(n_splits=5, shuffle=True, random_state=546789)
folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=90210) # 0 !!!!!
oof_preds = np.zeros(data.shape[0])
sub_preds = np.zeros(test.shape[0])

feature_importance_df = pd.DataFrame()
roc=pd.DataFrame(columns=['auc','fold'])
feats = [f for f in data.columns if f not in ['SK_ID_CURR']]
from scipy.stats import rankdata
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
        early_stopping_rounds=100,
        is_unbalance=False,
        learning_rate=0.02,
        max_bin=300,
        max_depth=-1,
        metric='auc',
        min_child_samples=70,
        min_split_gain=0.5,
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
    
    clf.fit(trn_x, trn_y, 
            eval_set= [(trn_x, trn_y), (val_x, val_y)], 
            eval_metric='auc', verbose=1000, early_stopping_rounds=100,categorical_feature=CATEGORICAL_COLUMNS #30
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
roc=roc.append({'auc':roc_auc_score(y, oof_preds),'fold':0},ignore_index=True)
roc.to_csv("../output/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time()+3600*7))+"_roc.csv", index=False)
feature_importance_df.to_csv("../output/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time()+3600*7))+"_importance.csv", index=False)
test['TARGET'] = sub_preds

from time import gmtime, strftime
test[['SK_ID_CURR', 'TARGET']].to_csv("../output/"+strftime("%Y-%m-%d_%H-%M-%S", gmtime(time()+3600*7))+"_first_submission.csv", index=False)
sys_exit(0)
# Plot feature importances
cols = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(
    by="importance", ascending=False)[:150].index

best_features = feature_importance_df.loc[feature_importance_df.feature.isin(cols)]

colstotal = feature_importance_df[["feature", "importance"]].groupby("feature").mean().sort_values(by="importance", ascending=False).index
feature_importance_df.loc[feature_importance_df.feature.isin(colstotal)].to_csv("../output/feature_importance.csv",index=False)

plt.figure(figsize=(20,20))
sns.barplot(x="importance", y="feature", data=best_features.sort_values(by="importance", ascending=False))
plt.title('LightGBM Features (avg over folds)')
#plt.legend(prop={'size':6})
plt.tight_layout()
plt.savefig('lgbm_importances.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
scores = [] 
for n_fold, (_, val_idx) in enumerate(folds.split(data,y)):  
    # Plot the roc curve
    fpr, tpr, thresholds = roc_curve(y.iloc[val_idx], oof_preds[val_idx])
    score = roc_auc_score(y.iloc[val_idx], oof_preds[val_idx])
    scores.append(score)
    plt.plot(fpr, tpr, lw=1, alpha=0.3, label='ROC fold %d (AUC = %0.4f)' % (n_fold + 1, score))

plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Luck', alpha=.8)
fpr, tpr, thresholds = roc_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(fpr, tpr, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('LightGBM ROC Curve')
plt.legend(loc="lower right")
#plt.tight_layout()

plt.savefig('roc_curve.png')

# Plot ROC curves
plt.figure(figsize=(6,6))
precision, recall, thresholds = precision_recall_curve(y, oof_preds)
score = roc_auc_score(y, oof_preds)
plt.plot(recall, precision, color='b',
         label='Avg ROC (AUC = %0.4f $\pm$ %0.4f)' % (score, np.std(scores)),
         lw=2, alpha=.8)

plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('LightGBM Recall / Precision')
plt.legend(loc="lower right")
#plt.tight_layout()

plt.savefig('recall_precision_curve.png')

