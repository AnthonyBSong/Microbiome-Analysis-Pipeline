#!/usr/bin/env python
# coding: utf-8

# In[57]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

try_df = pd.read_csv('try.csv')

percents_we_want = {'Escherichia_Shigella': [], 'Enterococcus': [], 'Akkermansia': [], 'Oscillibacter': [],  'Anaerotruncus' : [],
                   'Ruminococcus': [], 'Coprococcus': [], 'ClostridiumXVIII': [], 'Erysipelotrichaceae_incertae_sedis': [],
                   'Fusicatenibacter' : []}

for i in percents_we_want:
    lst = []
    for j in range(175):
        lst.append(try_df[i].iloc[j]/try_df['Total'].iloc[j])
    percents_we_want[i] = lst
    
#percents_we_want['Escherichia_Shigella']


# In[58]:


def Average(lst):
    return sum(lst) / len(lst)


# In[59]:


zeros = try_df.index[(try_df['clusters']==0)].tolist()
ones = try_df.index[(try_df['clusters']==1)].tolist()
twos = try_df.index[(try_df['clusters']==2)].tolist()


# In[66]:


a = 'Escherichia_Shigella'

lst = [item for item in percents_we_want[a]] 

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]



print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))

norm_1


# In[47]:


a = 'Enterococcus'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[48]:


a = 'Akkermansia'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[49]:


a = 'Oscillibacter'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[50]:


a = 'Anaerotruncus'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[51]:


a = 'Ruminococcus'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[52]:


a = 'Coprococcus'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[53]:


a = 'ClostridiumXVIII'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[54]:


a = 'Erysipelotrichaceae_incertae_sedis'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[55]:


a = 'Fusicatenibacter'

microbiota1 = [item for i, item in enumerate(percents_we_want[a]) if i in zeros]
microbiota2 = [item for i, item in enumerate(percents_we_want[a]) if i in ones]
microbiota3 = [item for i, item in enumerate(percents_we_want[a]) if i in twos]


print('Cluster 1 Percent Variation Explained for', a, ": ", Average(microbiota1))
print('Cluster 2 Percent Variation Explained for', a, ": ", Average(microbiota2))
print('Cluster 3 Percent Variation Explained for', a, ": ", Average(microbiota3))


# In[ ]:




