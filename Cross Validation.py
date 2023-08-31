#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd
from sklearn.metrics import accuracy_score


# In[19]:


import warnings
warnings.filterwarnings("ignore")


# In[20]:


df = pd.read_csv('place.csv')
X = df.iloc[:,1:3]
y = df.iloc[:,-1]
df


# # Train Test Split

# In[21]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# In[22]:


from sklearn.linear_model import LogisticRegression
logr = LogisticRegression()
logr.fit(X_train,y_train)
y_pred = logr.predict(X_test)

print("Test data accuracy:",accuracy_score(y_test, y_pred))


# # Cross Validation

# In[23]:


from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[24]:


# K-Fold
logr1=LogisticRegression()
score=cross_val_score(logr1,X,y,cv=5)

print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation (Test data accuracy): {}".format(score.mean()))


# In[25]:


#Stratified KFold is used for imbalanced data

logr2=LogisticRegression()
score=cross_val_score(logr2,X,y,cv= StratifiedKFold(5))

print("Cross Validation Scores are {}".format(score))
print("Average Cross Validation (Test data accuracy): {}".format(score.mean()))


# In[ ]:




