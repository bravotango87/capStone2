#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import pandas, matplotlib.pyplot, and seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np


# In[2]:


# upload credit table data

credit_record = pd.read_csv("credit_record.csv")

credit_record


# In[4]:


#renamed columns

credit_record.rename(columns = {'MONTHS_BALANCE':'Balance','STATUS':'Status'}, inplace=True)

credit_record


# In[5]:


# groupby ID & Balance

credit_record.groupby(['ID'])['Balance'].count()

credit_record


# In[6]:


# get mean 

credit_record.groupby(['ID'])['Balance'].mean()


# In[7]:


credit_record.groupby(['ID'])['Balance'].mean().reset_index()


# In[10]:


credit_record.groupby(['ID', 'Status'])['Balance'].mean()


# In[11]:


credit_record.groupby(['ID', 'Status'])['Balance'].mean().reset_index()


# In[12]:


credit_record.groupby(['ID', 'Status'])['Balance'].agg('mean').reset_index()


# In[14]:




credit_mean = credit_record.groupby(['ID','Status'], as_index= False).mean().pivot('ID','Status').fillna(0)


# In[15]:


credit_mean


# In[17]:


# credit min

credit_record.groupby(['ID'])['Balance'].min()


# In[18]:



credit_record.groupby(['ID'])['Balance'].min().reset_index()


# In[19]:


credit_record.groupby(['ID','Status'])['Balance'].min()


# In[20]:


credit_record.groupby(['ID','Status'])['Balance'].min().reset_index()


# In[21]:


credit_record.groupby(['ID','Status'])['Balance'].agg('min').reset_index()


# In[24]:


# pivot

credit_min = credit_record.groupby(['ID','Status'], as_index= False).min().pivot('ID','Status').fillna(0)

credit_min


# In[26]:


# get max

credit_record.groupby(['ID'])['Balance'].max()


# In[27]:


# reset index


credit_record.groupby(['ID'])['Balance'].max().reset_index()


# In[28]:


credit_record.groupby(['ID','Status'])['Balance'].max()


# In[29]:


credit_record.groupby(['ID','Status'])['Balance'].max().reset_index()


# In[30]:


credit_record.groupby(['ID','Status'])['Balance'].agg('max').reset_index()


# In[36]:


credit_max= credit_record.groupby(['ID','Status'],as_index=False).max().pivot('ID','Status').fillna(0)

credit_max


# In[ ]:


# need to figure out how to merge all 3 to get the min mean and max for the ID

