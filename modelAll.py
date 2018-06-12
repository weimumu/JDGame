# coding: utf-8
import pandas as pd
import numpy as np
from sklearn.cross_validation import *
import lightgbm as lgb
from sklearn import metrics
from sklearn.ensemble import GradientBoostingClassifier
import xgboost as xgb
from sklearn.grid_search import GridSearchCV

train = pd.read_csv('../data/train_featureV1.csv')
test = pd.read_csv('../data/test_featureV1.csv')

dtrain = lgb.Dataset(train.drop(['uid','label'],axis=1),label=train.label)
dtest = lgb.Dataset(test.drop(['uid'],axis=1))


# light 模型
lgb_params =  {
    'boosting_type': 'gbdt',
    'objective': 'binary',
#    'metric': ('multi_logloss', 'multi_error'),
    #'metric_freq': 100,
    'is_training_metric': False,
    'min_data_in_leaf': 12,
    'num_leaves': 64,
    'learning_rate': 0.08,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'verbosity':-1,
#    'gpu_device_id':2,
#    'device':'gpu'
#    'lambda_l1': 0.001,
#    'skip_drop': 0.95,
#    'max_drop' : 10
    #'lambda_l2': 0.005
    #'num_threads': 18
}
def evalMetric(preds, dtrain):
    label = dtrain.get_label()

    pre = pd.DataFrame({'preds': preds, 'label': label})
    pre = pre.sort_values(by='preds', ascending=False)

    auc = metrics.roc_auc_score(pre.label, pre.preds)

    pre.preds = pre.preds.map(lambda x: 1 if x >= 0.5 else 0)

    f1 = metrics.f1_score(pre.label, pre.preds)

    res = 0.6 * auc + 0.4 * f1
    return 'res', res, True

lgb.cv(lgb_params,dtrain,feval=evalMetric,early_stopping_rounds=100,verbose_eval=5,num_boost_round=10000,nfold=3,metrics=['evalMetric'])
model =lgb.train(lgb_params,dtrain,feval=evalMetric,verbose_eval=5,num_boost_round=300,valid_sets=[dtrain])
pred=model.predict(test.drop(['uid'],axis=1))
res =pd.DataFrame({'uid':test.uid,'label':pred})
print(res.head())


# GBDT模型

gsearch1 = GradientBoostingClassifier(learning_rate=0.05, n_estimators=80,max_depth=11, min_samples_leaf =60,
               min_samples_split =10, max_features=17, subsample=0.75, random_state=10)
gsearch1.fit(train.drop(['uid','label'],axis=1), train['label'])
GBDTpred = gsearch1.predict_proba(test.drop(['uid'],axis=1))[:,1]
GBDTres = pd.DataFrame({'uid': test.uid, 'label': GBDTpred})
print(GBDTres.head())


xgb_model = xgb.XGBClassifier()

parameters = {'nthread':[4], #when use hyperthread, xgboost may become slower
              'objective':['binary:logistic'],
              'learning_rate': [0.1], #so called `eta` value
              'max_depth': [6],
              'min_child_weight': [4],
              'silent': [1],
              'subsample': [0.7],
              'colsample_bytree': [1.0],
              'n_estimators': [85], #number of trees, change it to 1000 for better results
              'missing':[-999],
              'seed':[700]}

def evalMetric1(y, y_pred):
    return 0.4 * metrics.f1_score(y, y_pred) + 0.6 * metrics.roc_auc_score(y, y_pred)

myEvel = metrics.make_scorer(evalMetric1, greater_is_better=True)

clf = GridSearchCV(xgb_model, parameters, n_jobs=5,
                   cv=StratifiedKFold(train['label'], n_folds=5, shuffle=True),
                   scoring=myEvel,
                   verbose=2, refit=True)

clf.fit(train.drop(['uid','label'],axis=1), train['label'])

XGBPred = clf.predict_proba(test.drop(['uid'],axis=1))[:, 1]
XGBres = pd.DataFrame({'uid': test.uid, 'label': XGBPred})

print(XGBres.head())


result = pd.DataFrame({'uid': test.uid})
result['label'] = res['label'] + GBDTres['label'] + XGBres['label']


result=result.sort_values(by='label',ascending=False)
result.label=result.label.map(lambda x: 1 if x>=1.5 else 0)
result.to_csv('../result_B/all_model_result.csv',index=False,header=False,sep=',',columns=['uid','label'])
