#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder , OneHotEncoder , OrdinalEncoder
from sklearn.preprocessing import MinMaxScaler , StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib.inline', '')


# In[2]:


data=pd.read_csv('hr_company.csv')


# In[3]:


print(data.describe())


# In[4]:


print(data.info())


# In[5]:


data.isnull().sum()


# In[6]:


data.nunique()


# In[7]:


# splitting the data into features and targets
y=data['left']


# In[8]:


X=data.drop(['left'],axis=1)


# In[9]:


print(X.shape)
print(y.shape)


# In[10]:


# splitting the data into train and test 
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.33,random_state=7)


# In[11]:


# checking if train test is perform well or not
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[12]:


X_train


# In[13]:


# first dividing the categorical and numerical data
# 
X_train_num=X_train.drop(['sales','salary'],axis=1)
X_train_cat=X_train[['sales','salary']]
X_test_num=X_test.drop(['sales','salary'],axis=1)
X_test_cat=X_test[['sales','salary']]


# In[14]:


# using ordinal encoder for categoricl data
oe= OrdinalEncoder()
oe.fit(X_train_cat)
X_train_cat_enc=oe.transform(X_train_cat)


oe.fit(X_test_cat)
X_test_cat_enc=oe.transform(X_test_cat)


# In[15]:


# using standard scaler for numerical data

sc=StandardScaler()
sc.fit(X_train_num)
X_train_num_enc=sc.transform(X_train_num)

sc.fit(X_test_num)
X_test_num_enc=sc.transform(X_test_num)


# In[16]:


# using label encoder for target 

le=LabelEncoder()
le.fit(y_train)
y_train_enc=le.transform(y_train)
le.fit(y_train)
y_test_enc=le.transform(y_test)


# In[17]:


print(X_train_cat_enc)
print(X_test_cat_enc)


# In[18]:


print(X_train_num_enc)
print(X_test_num_enc)


# In[19]:


print(y_train_enc)


# In[20]:


# print(y_test_enc)
print(X_train_num_enc)
print(X_train_cat_enc)


# In[21]:


# now conconate the train and test categorical and numerical data
X_train_num_enc_df=pd.DataFrame(X_train_num_enc)
X_train_cat_enc_df=pd.DataFrame(X_train_cat_enc)

X_test_num_enc_df=pd.DataFrame(X_test_num_enc)
X_test_cat_enc_df=pd.DataFrame(X_test_cat_enc)



# In[22]:


X_train_cat_enc_df.rename(columns={0:7,1:8},inplace=True)
X_test_cat_enc_df.rename(columns={0:7,1:8},inplace=True)


# In[23]:


X_test_final=pd.concat([X_test_num_enc_df,X_test_cat_enc_df],axis=1)
X_train_final=pd.concat([X_train_num_enc_df,X_train_cat_enc_df],axis=1)


# In[24]:


print(X_test_final)
print(X_train_final)


# In[25]:


print(X_test_cat_enc_df)
print(X_train_cat_enc_df)


# In[ ]:





# In[26]:


X_train_final


# In[ ]:





# In[34]:


# model building 
X_train_final_values=X_train_final.values
model_final=LogisticRegression(solver='liblinear',class_weight={0:.9,1:.1})
model_final.fit(X_train_final,y_train_enc)
y_pred_final=model_final.predict(X_test_final)


# In[35]:


from sklearn.metrics import accuracy_score
ACCURACY_final=accuracy_score(y_pred_final,y_test_enc)


# In[36]:


print(ACCURACY_final)


# In[30]:


from sklearn.tree import DecisionTreeClassifier


# In[37]:


DTC=DecisionTreeClassifier()
DTC.fit(X_train_final,y_train_enc)
y_pred=DTC.predict(X_test_final)
ACCURAY=accuracy_score(y_pred,y_test_enc)
print(ACCURAY)


# In[38]:


# hyper[arameter tuning for logistic regression
from sklearn.model_selection import StratifiedKFold ,GridSearchCV

lr=LogisticRegression()
weights = np.linspace(0.0,0.99,500)
#specifying all hyperparameters with possible values
param= {'C': [0.1, 0.5, 1,10,15,20], 'penalty': ['l1', 'l2'],"class_weight":[{0:x ,1:1.0 -x} for x in weights]}
# create 5 folds
folds = StratifiedKFold(n_splits = 5, shuffle = True, random_state = 42)
#Gridsearch for hyperparam tuning
model1= GridSearchCV(estimator= lr,param_grid=param,scoring="f1",cv=folds,return_train_score=True)
#train model to learn relationships between x and y
model1.fit(X_train_final,y_train_enc)


# In[39]:


# print best hyperparameters
print("Best F1 score: ", model1.best_score_)
print("Best hyperparameters: ", model1.best_params_)


# In[40]:


lr1=LogisticRegression(C=20,class_weight={0: 0.317434869739479, 1: 0.682565130260521}, penalty= 'l2')
lr1.fit(X_train_final,y_train_enc)


# In[42]:


from sklearn.metrics import confusion_matrix,r2_score,recall_score,precision_score,roc_auc_score,f1_score


# In[47]:


y_pred_prob_test = lr1.predict_proba(X_test_final)[:, 1]
#predict labels on test dataset

y_pred_test = lr1.predict(X_test_final)
# create onfusion matrix
cm = confusion_matrix(y_test_enc, y_pred_test)
print("confusion Matrix is :nn",cm)
print("n")
# ROC- AUC score
print("ROC-AUC score  test dataset:  t", roc_auc_score(y_test_enc,y_pred_prob_test))
#Precision score
print("precision score  test dataset:  t", precision_score(y_test_enc,y_pred_test))
#Recall Score
print("Recall score  test dataset:  t", recall_score(y_test_enc,y_pred_test))
#f1 score
print("f1 score  test dataset :  t", f1_score(y_test_enc,y_pred_test))
# accuracy_score
print('accuracy score test dataset : t',accuracy_score(y_test_enc,y_pred_test))


# In[ ]:




