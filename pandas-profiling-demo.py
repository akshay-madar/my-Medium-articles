#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas_profiling
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# get sample data from sklearn
from sklearn.datasets import load_boston
boston = load_boston()


# In[3]:


# read a brief summary about the boston dataset
print("keys:",boston.keys())
print("shape:",boston.data.shape)
print("feature names:",boston.feature_names)
print("Description:",boston.DESCR)


# In[12]:


# converting to data frames
bos = pd.DataFrame(boston.data) # create the data frame
bos.columns = boston.feature_names # label columns
bos['PRICE'] = boston.target # Create price column


# In[13]:


# performing EDA using pandas
bos.describe()


# In[11]:


# performing EDA using pandas-profiling
profile = pandas_profiling.ProfileReport(bos)
profile


# In[ ]:




