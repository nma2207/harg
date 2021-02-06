#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[ ]:





# In[2]:


file_names = pd.read_csv("D:/hahaton/train.csv")
file_names.head()


# In[3]:


result = []
for i, r in file_names.iterrows():
    df = pd.read_csv("D:/hahaton/data_train/data_train/" + r["id"])
    df["category"] = r["category"]
    result.append(df)
result[0].tail()


# In[4]:


result = pd.concat(result)


# In[5]:


result.describe()


# In[9]:


plt.hist(result["H2"], bins=50)


# In[28]:


fig, axs = plt.subplots(1, 4)
n_bins = 100
fig.set_size_inches(18.5, 5.5)
axs[0].hist(result['H2'], bins=n_bins, density=True)
axs[0].set_title('H2')
axs[1].hist(result['CO'], bins=n_bins)
axs[1].set_title('H2')
axs[2].hist(result['C2H2'], bins=n_bins)
axs[2].set_title('c2h32')
axs[3].hist(result['C2H4'], bins=n_bins)
axs[3].set_title('c2h4')


# In[2]:


for k in range(1, 5):
    r = result[result["category"]==k]
    fig, axs = plt.subplots(1, 4)
    n_bins = 100
    fig.set_size_inches(18.5, 5.5)
    axs[0].hist(r['H2'], bins=n_bins)
    axs[0].set_title('H2')
    axs[1].hist(r['CO'], bins=n_bins)
    axs[1].set_title('H2')
    axs[2].hist(r['C2H2'], bins=n_bins)
    axs[2].set_title('c2h32')
    axs[3].hist(r['C2H4'], bins=n_bins)
    axs[3].set_title('c2h4')


# In[33]:


r = [result[result["category"]==k]["H2"] for k in range(1, 5)]


# In[38]:


plt.hist(r, bins = 100, stacked=True,
         label = [1, 2, 3, 4])
plt.legend()


# In[41]:


plt.figure()
plt.plot(result["CO"][:420])


# In[42]:


import statsmodels

