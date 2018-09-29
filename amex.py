
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import xgboost as xgb
import lightgbm as lgb
import gc

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import Imputer
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier

sns.set_style("dark")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

train_df = pd.read_csv("train.csv",header=None)
test_df = pd.read_csv("test.csv",header=None)

answer = pd.DataFrame()
answer['key'] = test_df.iloc[:,0]

train_df.drop(train_df.columns[0],axis=1,inplace=True,errors='ignore')
test_df.drop(test_df.columns[0],axis=1,inplace=True,errors='ignore')

train_x = train_df.iloc[:,:-1]
train_y = train_df.iloc[:,-1]


x_train, x_valid, y_train, y_valid = train_test_split(train_x,train_y,test_size=0.2,random_state=24)

xgb_dtrain = xgb.DMatrix(x_train, label=y_train)
xgb_dvalid = xgb.DMatrix(x_valid, label=y_valid)


xgb_params = {
    'boosting_type': 'gbdt',
    'objective': 'binary:logistic',
    'metric': 'auc',
    'learning_rate': 0.04,
    'verbose': 0,
    'num_leaves': 1024,
    'max_depth' : 16,
    'max_bin': 255,
    }

evals_results = {}
print("Training the xgb model...")
watchlist = [(xgb_dtrain, 'train'),(xgb_dvalid, 'valid')]

xgb_model = xgb.train(xgb_params, 
                 xgb_dtrain,
                 4500,
                 watchlist,
                 evals_result=evals_results, 
                 early_stopping_rounds=70,
                 verbose_eval=True)


xgb_x = xgb_model.predict(xgb.DMatrix(test_df),ntree_limit=xgb_model.best_iteration)

answer['score'] = xgb_x
fig, ax = plt.subplots(figsize=(12,18))
xgb.plot_importance(xgb_model, max_num_features=50, height=0.8, ax=ax)
plt.show()
train_df = train_df.rename(columns=lambda x: str(x))
sns.violinplot(x="10",y="55",data=train_df.rename(columns=lambda x: str(x)))

train_df.groupby(by=['55','1'])
answer.to_csv('submission_xgb_4500.csv',index=False)

train_df.columns
lgb_params = {
    'boosting_type': 'dart',
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.02,
    'verbose': 0,
    'num_leaves': 1024,  
    'max_depth': 16,  
    'max_bin': 255,
    'num_threads' : 2
    }

lgb_score4 = pd.DataFrame(columns=range(0,4))
#lgb_score.iloc[:,count] = pd.Series(test_df.iloc[:,0])
count = 0

for state in range(25,96,20):
    print(state)
    x_train, x_valid, y_train, y_valid = train_test_split(train_x,train_y,test_size=0.25,random_state=state)

    dtrain = lgb.Dataset( x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid)

    evals_results = {}
    print("Training the model...")
    
    lgb_model = lgb.train(lgb_params, 
                     dtrain, 
                     valid_sets=[dtrain, dvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=7000,
                     early_stopping_rounds=80,
                     verbose_eval=True, 
                     feval=None)

    lgb_x = lgb_model.predict(test_df, num_iteration=lgb_model.best_iteration)
    lgb_score4.iloc[:,count] = pd.Series(lgb_x)
    count += 1
    
lgb_score2 = pd.DataFrame(columns=range(0,3))
count = 0

for size in np.arange(0.15,0.29,0.05):
    x_train, x_valid, y_train, y_valid = train_test_split(train_x,train_y,test_size=size,random_state=40)

    dtrain = lgb.Dataset(x_train, label=y_train)
    dvalid = lgb.Dataset(x_valid, label=y_valid)

    evals_results = {}
    print("Training the model...")
    
    lgb_model = lgb.train(lgb_params, 
                     dtrain, 
                     valid_sets=[dtrain, dvalid], 
                     valid_names=['train','valid'], 
                     evals_result=evals_results, 
                     num_boost_round=5000,
                     early_stopping_rounds=80,
                     verbose_eval=True, 
                     feval=None)

    lgb_x = lgb_model.predict(test_df, num_iteration=lgb_model.best_iteration)
    lgb_score2.iloc[:,count] = pd.Series(lgb_x)
    count += 1

fold_importance_df = pd.DataFrame()
fold_importance_df["feature"] = train_x.columns
fold_importance_df["importance"] = lgb_model.feature_importance()
plt.figure(figsize=(18,20))
sns.barplot(x='importance',y='feature',data=fold_importance_df.sort_values(by="importance", ascending=False))

sum1 = lgb_score2.sum(axis=1)/3
sum2 = lgb_score.sum(axis=1)/5
sum4 = lgb_score4.sum(axis=1)/6
answer['score'] = 0.75*sum4 + 0.25*sum2

answer.to_csv('submission_lgb_dart_3_5_mixed_final.csv',index=False)
