#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
pd.set_option('display.max_columns', 2000)
pd.set_option('display.max_rows', 2000)
from sklearn.metrics import accuracy_score,f1_score,auc,roc_auc_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[2]:


data=pd.read_csv("merge.txt",encoding='utf-8',sep='\t',header=None)


# In[3]:


data.columns=[
'features_needed',
  'label_1',
  'label_2'  
]


# In[4]:


#drop columns
data.drop('nums_le_15', axis=1, inplace=True)
#clean data with \\N but not NaN
data = data.replace('\\N', np.nan)
#drop NA values
data = data.dropna()
#count NA values
data.isna().sum()


# In[7]:


#transform features
lb=LabelEncoder()
data['type1']=lb.fit_transform(data['type1'])
data['type2']=lb.fit_transform(data['type2'])
data['type3']=lb.fit_transform(data['type3'])


# In[9]:


#check sample datas
data=data.sample(frac=1.0)


# In[10]:


data


# In[11]:


#split test and train datas
train_data=data[:int(len(data)*0.8)]


# In[12]:


test_data=data[int(len(data)*0.8):]


# In[13]:


#check sizes
len(train_data),len(test_data)


# ## LGB label_1

# In[14]:


import lightgbm as lgb


# In[15]:


train_data['upload_diff']=train_data['upload_diff'].astype(int)


# In[16]:


#select features needed
X=train_data[['feature1', 'feature2','type1','type2','type3']]
y=train_data[['label_1']]


# In[18]:


X_train,X_test, y_train, y_test =train_test_split(X,y,test_size=0.1, random_state=0)


# In[33]:


lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_test, y_test, reference=lgb_train)
# specify your configurations as a dict
params = {
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': {'binary_logloss', 'auc'},
    'num_leaves': 5,
    'max_depth': 6,
    'min_data_in_leaf': 450,
    'learning_rate': 0.1,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.95,
    'bagging_freq': 5,
    'lambda_l1': 1,  
    'lambda_l2': 0.001,  
    'min_gain_to_split': 0.2,
    'verbose': 5,
    'is_unbalance': True
}


# In[34]:


print('Start training...')
gbm = lgb.train(params,
                lgb_train,
                num_boost_round=200,
                valid_sets=lgb_eval,
                early_stopping_rounds=10)


# In[35]:


test_data['diff']=test_data['diff'].astype(int)


# In[36]:


test_=test_data[['feature1', 'feature2','type1','type2','type3']]


# In[37]:


test_label=test_data['label_1']


# In[38]:


preds = gbm.predict(test_, num_iteration=gbm.best_iteration)


# In[39]:


preds_int=[1 if item>0.5 else 0 for item in preds]


# In[40]:


accuracy_score(y_pred=preds_int,y_true=test_label),f1_score(y_pred=preds_int,y_true=test_label),roc_auc_score(y_score=preds_int,y_true=test_label)


# In[80]:


lgb_predictors=[item for item in test_.columns]


# In[81]:


lgb_feat_imp = pd.Series(gbm.feature_importance(), lgb_predictors).sort_values(ascending=False)


# In[ ]:


#feature importance
lgb_feat_imp


# In[ ]:




