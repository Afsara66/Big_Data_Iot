#!/usr/bin/env python
# coding: utf-8

# In[78]:


import pandas as pd


# In[79]:


dataset = ['Hello my name is james',
          'james this is my python notebook named james',
          'james trying to create a big dataset',
          'james of words to try differnt',
          'features of count vectorizer']


# In[80]:


dataset


# # Count Vectorizer

# In[81]:


from sklearn.feature_extraction.text import CountVectorizer


# In[82]:


cv=CountVectorizer(lowercase=False)


# In[83]:


x=cv.fit_transform(dataset)


# In[84]:


feature_name= cv.get_feature_names()  #features of the dataset
feature_name


# In[85]:


cv.vocabulary_ #position of the words in the matrix


# In[86]:


count_array= x.toarray() 
count_array


# In[87]:


df = pd.DataFrame(data=count_array,columns = feature_name) #sparse matrix of dataset


# In[88]:


df.shape


# In[89]:


df


# # TF-IDF Vectorizer 

# In[90]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[91]:


cv=TfidfVectorizer()


# In[92]:


x=cv.fit_transform(dataset)


# In[93]:


feature_name= cv.get_feature_names()  #features of the dataset
feature_name


# In[94]:


cv.vocabulary_ #position of the words in the matrix


# In[95]:


count_array= x.toarray() 
count_array


# In[96]:


df = pd.DataFrame(data=count_array,columns = feature_name) #sparse matrix of dataset
df.shape


# In[97]:


df


# # Spam Email Detection using Vectorizer

# In[98]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn
from sklearn import preprocessing
from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
get_ipython().run_line_magic('matplotlib', 'inline')


# In[99]:


df = pd.read_csv('C:/Users/Administrator/Downloads/ML/Data/emails.csv')


# In[100]:


df


# In[101]:


df.shape


# In[102]:


df['spam'].value_counts()


# In[103]:


seaborn.countplot(x='spam',data=df)


# In[104]:


df.isnull().sum()


# In[105]:


X= df.text.values
y= df.spam.values


# In[106]:


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()
X_vectorized=cv.fit_transform(X)
X_vectorized.toarray()


# In[107]:


#Dataset splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_vectorized,y,test_size=.25,random_state=1)


# In[108]:


from sklearn.naive_bayes import MultinomialNB

#Create a Gaussian Classifier
mnb = MultinomialNB()
mnb.fit(X_train, y_train)
pred=mnb.predict(X_test)


# In[109]:


print("Accuracy score: ", accuracy_score(y_test,pred))


# In[110]:


confusion_matrix(y_test,pred)


# In[111]:


print(classification_report(y_test,pred))


# In[112]:


class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
seaborn.heatmap(pd.DataFrame(confusion_matrix(y_test,pred)), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')


# In[ ]:




