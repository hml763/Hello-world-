import os
import pandas as pd
import numpy as np
import datetime
import lightgbm as lgb
from sklearn.model_selection import train_test_split

#os.chdir("C:/Users/hml76/OneDrive/바탕 화면/")
#train_raw = pd.read_csv("application_train.csv")
#test_raw = pd.read_csv("application_test.csv")

dataset = np.loadtxt('C:/Users/hml76/OneDrive/바탕 화면/KDN참가신청서류_김두진 외 3명/1221.xlsx', delimiter=',', encoding='ANSI')
#dataset = np.loadtxt('C:/Users/hml76/OneDrive/바탕 화면/2019학년도 4학년1학기/지능시스템/과제/pima-indians-diabetes.data_utf.csv', delimiter=',', encoding='cp949')


train, validation = train_test_split(train_raw, test_size = 0.2)

y_train = train['TARGET']
x_train = train.drop(['TARGET'], axis = 1)
y_val = validation['TARGET']
x_val = validation.drop(['TARGET'], axis = 1)

x_train2 = x_train.drop(['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'], axis=1)
x_val2 = x_val.drop(['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'], axis=1)

x_test2 = test_raw.drop(['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','NAME_TYPE_SUITE','NAME_INCOME_TYPE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS','NAME_HOUSING_TYPE','OCCUPATION_TYPE','WEEKDAY_APPR_PROCESS_START','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE','WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE'], axis=1)

train_ds = lgb.Dataset(x_train2, label = y_train)
val_ds = lgb.Dataset(x_val2, label = y_val)

params = {'learning_rate': 0.01, 'max_depth': 16, 'boosting': 'gbdt', 'objective': 'regression', 'metric': 'auc', 'is_training_metric': True, 'num_leaves': 144, 'feature_fraction': 0.9, 'bagging_fraction': 0.7, 'bagging_freq': 5, 'seed':2018}

model = lgb.train(params, train_ds, 1000, val_ds, verbose_eval=10, early_stopping_rounds=100)

submission = pd.DataFrame()
submission['SK_ID_CURR'] = test_raw['SK_ID_CURR']
submission['TARGET'] = model.predict(x_test2)
submission.to_csv('submission' + datetime.datetime.now().strftime("%I%M%p") + '.csv', index=False)

params = {'learning_rate': 0.01,
          'max_depth': 16,
          'boosting': 'gbdt',
          'objective': 'regression',
          'metric': 'auc',
          'is_training_metric': True,
          'num_leaves': 144,
          'feature_fraction': 0.9,
          'bagging_fraction': 0.7,
          'bagging_freq': 5,
          'seed':2018}

