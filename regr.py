#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os


# In[36]:


data = []
for file in os.listdir("D:/hahaton/time/data_test/data_test"):
    df_test = pd.read_csv("D:/hahaton/time/data_test/data_test/"+file)
    data.append(df_test)


# In[ ]:





# In[65]:


reg = GradientBoostingRegressor()
reg = LinearRegression()


# In[167]:


limit = {
    "H2":0.012,
    "C2H4":0.050,
    "C2H2":0.002,
    "CO":0.075,
}


# In[107]:


from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[168]:


print(limit)
result = []
for z, d in enumerate(data):
    #print(z)
    
    x = [[i,] for i in range(len(d))]
    times = []
    for gas in ["H2","C2H2", "C2H4", "CO"]:
        reg=GradientBoostingRegressor()
        reg.fit(x, d[gas])
        k = 0
        concent = reg.predict([[k,]])[0]
        while concent < limit[gas]:
            concent = reg.predict([[k,]])[0]
            k+=1
            if k > 2000:
                break
        times.append(k)
    result.append(min(times))


# In[ ]:





# In[169]:


result


# In[163]:


df = pd.DataFrame()


# In[164]:


df["id"] = os.listdir("D:/hahaton/time/data_test/data_test")
df["predicted"] = result


# In[165]:


df["predicted"]


# In[166]:


df.to_csv("D:/hahaton/time/test.csv", index=False)


# In[110]:


plt.plot(reg.predict([[i] for i in range(500)]))


# In[103]:


reg


# In[117]:


filename = "D:/hahaton/time/data_train/data_train/2_trans_497.csv"
limit = 550
reg=make_pipeline(PolynomialFeatures(4),LinearRegression())


# In[118]:


d = pd.read_csv(filename)


# In[120]:


for gas in ["H2","C2H2", "C2H4", "CO"]:
    reg.fit([[i,] for i in range(len(d))], d[gas])
    print(gas, reg.predict([[limit]]))


# In[ ]:





# In[122]:


file_names = pd.read_csv("D:/hahaton/time/train.csv")
file_names.head()


# In[149]:


result = []
concents = {
    "H2":[],
    "C2H2":[],
    "C2H4":[],
    "CO":[]
}

for i, r in file_names.iterrows():
    d = pd.read_csv("D:/hahaton/data_train/data_train/" + r["id"])
    limit = r["predicted"]
    for gas in ["H2","C2H2", "C2H4", "CO"]:
        reg=GradientBoostingRegressor()
        reg.fit([[i,] for i in range(len(d))], d[gas])
        
        concents[gas].append(reg.predict([[limit]])[0])


# In[151]:


for gas in ["H2","C2H2", "C2H4", "CO"]:
    #l = list(filter(lambda x : x > 0, concents[gas]))
    l = concents[gas]
    print(gas, min(l), max(l), sum(l)/len(l))


# In[158]:


limit = {
    "H2":0.0024433124693314977,
    "C2H4":0.00027144348728993216,
    "C2H2":0.0067553954042483925,
    "CO":0.023882999076160413,
}


# In[126]:


concents["H2"]

