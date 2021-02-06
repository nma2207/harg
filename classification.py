#!/usr/bin/env python
# coding: utf-8

# In[34]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[35]:


file_names = pd.read_csv("D:/hahaton/train.csv")
file_names.head()


# In[36]:


result = []
for i, r in file_names.iterrows():
    df = pd.read_csv("D:/hahaton/data_train/data_train/" + r["id"])
    df["category"] = r["category"]
    result.append(df)
result[0].tail()


# In[37]:


result = pd.concat(result)


# In[38]:


result


# In[11]:


from sklearn.ensemble import GradientBoostingClassifier


# In[12]:


clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1, max_depth=1, random_state=2)


# In[13]:


y = result.pop('category')
X = result


# In[129]:


X


# In[10]:


y


# In[96]:


def normalize(X):
    means = []
    stds = []
    for name in ("H2", "CO","C2H4","C2H2"):
        m = np.mean(X[name])
        std = np.std(X[name])
        X[name] = (X[name] - m) / std
        print(m)
        means.append(m)
        stds.append(std)
    return X, means, stds


# In[97]:


res, m, s = normalize(X)


# In[98]:


print(res, m, s)


# In[16]:


from sklearn.model_selection import train_test_split


# In[17]:


X_train,X_test,y_train,y_test = train_test_split(X.index,y,test_size=0.2)
train = X.iloc[X_train]
test = X.iloc[X_test]


# In[45]:


y_train


# In[109]:


w = np.zeros(len(y_train))
w[y_train==1] =  1. / sum(y_train==1)
w[y_train==2] = 1. / sum(y_train==2)
w[y_train==3] = 1. / sum(y_train==3)
w[y_train==4] = 1. / sum(y_train==4)


# In[39]:


print(sum(result["category"]==1))
print(sum(result["category"]==2))
print(sum(result["category"]==3))
print(sum(result["category"]==4))


# In[110]:


w


# In[200]:


train


# In[18]:


clf.fit(train, y_train)


# In[19]:


# 1 попытка
clf.score(test, y_test)


# In[24]:


class Logic:
    def load_clf(self):
        filename = "D:/hahaton/clf.sav"
        self.clf = pickle.load(open(filename, 'rb'))
    
    def get_defect(self, values)
        return self.clf.predict(values).tolist()
    


# In[25]:


l = Logic()
l.load_clf()


# In[33]:


l.clf.predict([[0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 1]]).tolist()


# In[20]:


import pickle


# In[21]:


pickle.dump(clf, open("D:/hahaton/clf.sav", 'wb'))


# In[18]:


from sklearn import linear_model
reg = linear_model.Ridge(alpha=.5)


# In[ ]:


reg.fit(train, y_train)


# In[55]:


df_test = pd.read_csv("D:/hahaton/data_test/data_test/2_trans_300.csv")
clf.predict(df_test)


# In[54]:


import os


# In[58]:


os.listdir("D:/hahaton/data_test/data_test/")


# In[203]:


test_res = []
for file in os.listdir("D:/hahaton/data_test/data_test/"):
    df_test = pd.read_csv("D:/hahaton/data_test/data_test/"+file)
    r = int(sum(clf.predict(df_test))/len(df_test))
    #r = clf.predict(df_test.iloc[[-1]])
    print(r)
    test_res.append(r)


# In[ ]:





# In[204]:


df = pd.DataFrame()
df["id"] = os.listdir("D:/hahaton/data_test/data_test/")
df["category"] = test_res


# In[205]:


df


# In[206]:


df.to_csv("D:/hahaton/test.csv", index=False)


# In[65]:


result.describe()


# In[61]:


len(os.listdir("D:/hahaton/data_test/data_test/"))


# In[57]:


from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier()


# In[63]:


sgd_clf.fit(train, y_train, sample_weight=w)


# In[65]:


sgd_clf.score(test, y_test)


# In[62]:





# In[6]:


import imblearn


# In[7]:


from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler


# In[14]:


#oversample = SMOTE()
undersample = RandomUnderSampler(sampling_strategy='majority')
#oversample = RandomOverSampler(sampling_strategy='minority')
#X, y = oversample.fit_resample(X, y)
X, y = undersample.fit_resample(X, y)


# In[216]:


X

