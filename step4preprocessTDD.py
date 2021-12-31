#!/usr/bin/env python
# coding: utf-8

# In[26]:


import pandas as pd
import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import __version__ as sklearn_version
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_regression
import datetime


# In[27]:


# supress warnings from plotnine

import warnings
warnings.filterwarnings('ignore')

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# load data

df2 = pd.read_csv("df1.csv")
df2.head(10)


# In[4]:


# select object columns

dfo = df2.select_dtypes(include=['object'])

df2 = pd.concat([df2.drop(dfo, axis=1), pd.get_dummies(dfo)], axis=1)

# objects converted to boolean

df2


# In[5]:


df2.drop(['MONTHS_BALANCE', 'open_month', 'end_month'], axis=1, inplace=True)
df2


# In[7]:


#Split the variables

X = df2.iloc[:, :2].values
y = df2.iloc[:, -1].values


# In[28]:


from sklearn.model_selection import train_test_split


# Splitting the dataset into the Training set and Test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 2718)


# In[29]:


print(X_train.shape)
print(X_test.shape)


# In[30]:


# Feature Scaling


sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


# In[31]:


# Fitting Logistic Regression to the Training set

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state =2718)
classifier.fit(X_train, y_train)


# In[32]:


# Predicting the Test set results

y_pred = classifier.predict(X_test)
y_pred


# In[34]:


# Making the Confusion Matrix


cm = confusion_matrix(y_test, y_pred)
acc_score = accuracy_score(y_test, y_pred)
print(acc_score*100)


# In[36]:


df2['STATUS'].value_counts(normalize=True)


# In[37]:


get_ipython().system('pip install imbalanced-learn')


# In[38]:


# check version number

import imblearn
print(imblearn.__version__)


# In[39]:


# determine which model performs the best

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[40]:


classifiers = {
    "LogisticRegression" : LogisticRegression(),
    "KNeighbors" : KNeighborsClassifier(),
    "SVC" : SVC(),
    "DecisionTree" : DecisionTreeClassifier(),
    "RandomForest" : RandomForestClassifier()
    
}


# In[43]:


# Train/Test Split

len(df2) * .7, len(df2) * .3


# In[44]:


# define min max scaler

scaler = MinMaxScaler()

# transform data

scaled = scaler.fit_transform(df2)

print(df2)


# In[45]:


df2


# In[46]:


# define standard scaler

scaler = StandardScaler()

# transform data

scaled = scaler.fit_transform(df2)
print(df2)


# In[47]:


from matplotlib import pyplot

# summarize the shape of the dataset
print(df2.shape)

# summarize each variable
print(df2.describe())

# histograms of the variables
df2.hist()
pyplot.show()


# In[48]:


from sklearn import preprocessing

scaler = preprocessing.StandardScaler().fit(X_train)
scaler


scaler.mean_


scaler.scale_


X_scaled = scaler.transform(X_train)
X_scaled


# In[49]:


X_scaled.mean(axis=0)


X_scaled.std(axis=0)


# In[50]:


# Making a Scaler object

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

 # apply scaling on training data
X, y = make_classification(random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
pipe = make_pipeline(StandardScaler(), LogisticRegression())
pipe.fit(X_train, y_train) 


# apply scaling on testing data, without leaking training data.
pipe.score(X_test, y_test)  


# In[51]:


# Fitting data to the scaler object

scaler = StandardScaler()
print(scaler.fit(df2))


# In[52]:


print(scaler.mean_)


# In[ ]:




