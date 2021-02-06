#!/usr/bin/env python
# coding: utf-8

# In[1]:


from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense


# In[9]:


import pandas as pd
import numpy as np
import matplotlib as plt
from numpy import array


# In[5]:


file_names = pd.read_csv("D:/hahaton/time/train.csv")
file_names.head()


# In[59]:


data = []

for i, r in file_names.iterrows():
    d = pd.read_csv("D:/hahaton/time/data_train/data_train/" + r["id"])
    limit = r["predicted"]
    predicted = list(range(len(d)+limit, limit, -1))
    d["predicted"] = predicted
    data.append(d)


# In[8]:


def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)


# In[60]:


data = pd.concat(data)


# In[29]:


from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# In[31]:





# In[32]:


data


# In[61]:


y = data.pop("predicted")
X = data


# In[35]:


from sklearn.model_selection import train_test_split


# In[62]:


X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
train = X.iloc[X_train]
test = X.iloc[X_test]


# In[69]:


reg = make_pipeline(LinearRegression())
reg.fit(train, y_train)


# In[42]:


X


# In[70]:


reg.score(test, y_test)


# In[19]:


model = Sequential()
model.add(LSTM(5, activation='relu', return_sequences=True, input_shape=(4, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')


# In[7]:


data = pd.concat(data)


# In[45]:


y =data[0].pop("predicted")
x = data[0]


# In[47]:


k = reg.predict(x)
k


# In[49]:


import os


# In[72]:


data = []
for file in os.listdir("D:/hahaton/time/data_test/data_test"):
    df_test = pd.read_csv("D:/hahaton/time/data_test/data_test/"+file)
    data.append(df_test.iloc[-1])


# In[74]:


res = list(map(abs, reg.predict(data)))


# In[75]:


res


# In[25]:


x = x.to_numpy()
y=y.to_numpy()


# In[22]:


x


# In[26]:


model.fit(x, y, epochs=10, verbose=0)


# In[10]:


split_sequence(data, 420)

