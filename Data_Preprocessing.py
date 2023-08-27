#!/usr/bin/env python
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np


# In[16]:


data = pd.read_csv('microbiome_ivsa.csv')
df_cleaned = pd.DataFrame()
feces = data.index[(data['Valid_Addiction_Index']==1)].tolist()

for i in range(len(feces)):
    df_cleaned = df_cleaned.append(data.iloc[feces[i],:])

colnames = list(data.columns[1:-1])


# In[17]:


df_cleaned.info()
df_cleaned.to_csv('df_cleaned.csv', index=False)


# In[18]:


microbiome = df_cleaned.iloc[:, 1:97]
behaviors = df_cleaned.iloc[:, 172:181] 
df_concat = pd.concat([microbiome, behaviors], axis=1)
df_concat.head()

microbiome_not_norm = pd.DataFrame()

for i in range(len(microbiome)):
    microbiome_not_norm = microbiome_not_norm.append(microbiome.iloc[i,:])


# In[19]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp.fit(df_concat)
temp_df = imp.transform(df_concat)
df_concat = pd.DataFrame(temp_df, index=df_concat.index, columns=df_concat.columns)


# In[20]:


whitelist=[]
if True:
    for col in df_concat.columns:
        if df_concat[col].std()!=0 and col not in whitelist:
            df_concat.loc[:,col] = (df_concat[col]-df_concat[col].mean())/df_concat[col].std()
        else:
            df_concat.loc[:,col] = 0
else:
    for col in df_concat.columns:
        if df_concat[col].max()-df_concat[col].min()!=0 and col not in whitelist:
            df_concat.loc[:,col] = (df_concat[col]-df_concat[col].min())/(df_concat[col].max()-df_concat[col].min())


# In[21]:


behaviors = df_concat.iloc[:, 96:106]
behaviors.to_csv('behaviour_new.csv', index=False)

microbiome_norm = df_concat.iloc[:, 0:96]

microbiome_norm


# In[22]:


normalized_microbiome = pd.DataFrame()
tossed_out = pd.DataFrame()
for col in microbiome_norm.columns:
    if not all(microbiome_norm[col] == 0):
        normalized_microbiome[col] = microbiome_norm[col]
    else:
        tossed_out[col] = microbiome_norm[col]


# In[23]:


non_normalized_microbiome = pd.DataFrame()
tossed_out2 = pd.DataFrame()
for col in microbiome_not_norm.columns:
    if not all(microbiome_not_norm[col] == 0):
        non_normalized_microbiome[col] = microbiome_not_norm[col]
    else:
        tossed_out2[col] = microbiome_not_norm[col]


# In[24]:


#to be removed
a = tossed_out
b = tossed_out2

#to be kept
c = normalized_microbiome
d = non_normalized_microbiome

set(a) ^ set(b)

set(c) ^ set(d)


# In[25]:


normalized_microbiome


# In[26]:


non_normalized_microbiome


# In[27]:


normalized_microbiome.to_csv('normalized.csv', index=False)


# In[28]:


non_normalized_microbiome.to_csv('non-normalized.csv', index=False)

