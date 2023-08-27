#!/usr/bin/env python
# coding: utf-8

# In[48]:


from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

# Dimension reduction and clustering libraries
import umap
import sklearn.cluster as cluster
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score

from sklearn.cluster import KMeans
from collections import Counter
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from sklearn.decomposition import PCA

import pandas as pd
import numpy as np


# In[49]:


behaviors = pd.read_csv('behaviour_new.csv')


# In[50]:


kmeans = KMeans(n_clusters=4, init='k-means++', random_state=0).fit(behaviors)
labels_2d = kmeans.labels_

score = silhouette_score(clusterable_embedding, labels_2d)
print('Silhouette Score: %.3f' % score)


# In[51]:


pca = PCA(n_components = 2)
clusterable_embedding = pca.fit_transform(behaviors)

kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(clusterable_embedding)
labels_2d = kmeans.labels_

score = silhouette_score(clusterable_embedding, labels_2d)
print('Silhouette Score: %.3f' % score)

sns.scatterplot(data=clusterable_embedding, x=clusterable_embedding[:, 0], y=clusterable_embedding[:, 1], hue=kmeans.labels_)
plt.show()


# In[52]:


# 2D Plot for Clustering
clusterable_embedding = umap.UMAP(
    n_neighbors=30,
    min_dist=0.0,
    metric='cosine',
    n_components=2,
    random_state=60,
).fit_transform(behaviors)

#print(clusterable_embedding)
kmeans = KMeans(n_clusters=2, init='k-means++', random_state=0).fit(clusterable_embedding)
#print(kmeans.labels_)  # same thing as kmeans.predict()
print(kmeans.inertia_)
#print(kmeans.n_iter_)
#print(kmeans.cluster_centers_)
#print(Counter(kmeans.labels_))

labels_2d = kmeans.labels_

sns.scatterplot(data=clusterable_embedding, x=clusterable_embedding[:, 0], y=clusterable_embedding[:, 1], hue=kmeans.labels_)
plt.show()

score = silhouette_score(clusterable_embedding, labels_2d)
print('Silhouetter Score: %.3f' % score)


# In[53]:


# 3D Plot for Clustering
from mpl_toolkits.mplot3d import Axes3D

clusterable_embedding = umap.UMAP(
    n_neighbors=7, #7
    min_dist=0.0, #0.0
    spread=1.0, #1.0
    set_op_mix_ratio=1.0, #1.0
    metric='euclidean', #euclidean
    n_components=3, #3
    random_state=20, #20
).fit_transform(behaviors)

#print(clusterable_embedding)
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=0).fit(clusterable_embedding)
UMAP_kmeans = kmeans.labels_
print(kmeans.labels_)  # same thing as kmeans.predict()
print(kmeans.inertia_)
print(kmeans.n_iter_)
centroids = kmeans.cluster_centers_
print(Counter(kmeans.labels_))

fig = plt.figure(figsize=(10,10))
ax = fig.gca(projection='3d')
plt.yticks(fontsize=18)
plt.xticks(fontsize=18)
ax.zaxis.set_tick_params(labelsize=18)

x = np.array(kmeans.labels_==0)
y = np.array(kmeans.labels_==1)
z = np.array(kmeans.labels_==2)

labels_3d = kmeans.labels_

ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c="black",s=150,label="Centers",alpha=1)
ax.scatter(clusterable_embedding[x,0],clusterable_embedding[x,1],clusterable_embedding[x,2],c="blue",s=40,label="C1")
ax.scatter(clusterable_embedding[y,0],clusterable_embedding[y,1],clusterable_embedding[y,2],c="green",s=40,label="C2")
ax.scatter(clusterable_embedding[z,0],clusterable_embedding[z,1],clusterable_embedding[z,2],c="red",s=40,label="C3")

score = silhouette_score(clusterable_embedding, labels_3d, metric='cosine')
print('Silhouetter Score: %.3f' % score)


# In[54]:


indices_1 = [ind for ind, ele in enumerate(labels_2d) if ele == 2]
indices_2 = [ind for ind, ele in enumerate(labels_3d) if ele == 2]

a = set(indices_1)
b = set(indices_2)

print(len(indices_1), len(indices_2))
print(len(a.intersection(b)))

#print("2D: {} \n\n3D:{}".format(indices_1, indices_2))


# In[55]:


print(clusterable_embedding)


# In[56]:


get_ipython().system('mkdir figs')


# In[57]:


# 3D 360 Degree Model for Cluster Analysis
# https://www.youtube.com/watch?v=XzAryDJTi1M&ab_channel=StatQuestwithJoshStarmer
for i in range(0, 360, 2):
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(projection='3d')

    ax.scatter(clusterable_embedding[x,0],clusterable_embedding[x,1],clusterable_embedding[x,2],c="blue",s=40,label="C1")
    ax.scatter(clusterable_embedding[y,0],clusterable_embedding[y,1],clusterable_embedding[y,2],c="green",s=40,label="C2")
    ax.scatter(clusterable_embedding[z,0],clusterable_embedding[z,1],clusterable_embedding[z,2],c="purple",s=40,label="C3")
    #ax.scatter(centroids[:,0],centroids[:,1],centroids[:,2],c="black",marker="^",s=150,label="Centers",alpha=1)

    x_center = (clusterable_embedding[:,0].max() + clusterable_embedding[:,0].min())/2
    y_center = (clusterable_embedding[:,1].max() + clusterable_embedding[:,1].min())/2
    z_center = (clusterable_embedding[:,2].max() + clusterable_embedding[:,2].min())/2

    #print(x_center, y_center, z_center)

    ax.plot([x_center, x_center], [y_center, y_center], [clusterable_embedding[:,2].min() - 2, clusterable_embedding[:,2].max() + 2],
       c='k', lw=1)
    ax.plot([x_center, x_center], [clusterable_embedding[:,1].min() - 2, clusterable_embedding[:,1].max() + 2], [z_center, z_center],
       c='k', lw=1)
    ax.plot([clusterable_embedding[:,0].min() - 2, clusterable_embedding[:,0].max() + 2], [y_center, y_center], [z_center, z_center],
       c='k', lw=1)

    ax.view_init(15, i)

    ax.axis("off")

    #plt.savefig(f'figs1/{i:003}.png', dpi=100, facecolor = 'white')
    
    plt.show()


# In[80]:


IVSA_cluster_means = pd.read_csv("behaviors.csv")
IVSA_cluster_means.info()


# In[81]:


IVSA_cluster_means = IVSA_cluster_means[['AQ_SessionsToAcquisition', 'DR_StS_1p0mgkg', 'DR_Inf_Total_AUC',
                                         'DR_Inf_Total_1p0mgkg', 'DR_Inf_Total_0p32mgkg', 'DR_Inf_Total_0p1mgkg', 
                                         'DR_Inf_Total_0p032mgkg', 'EX_ALP_Total_s02', 'RI_vs_Sal_ALP_Total_RIn_s01']]

IVSA_cluster_means.rename(columns={'AQ_SessionsToAcquisition': 'B1', 'DR_StS_1p0mgkg': 'B2',
                  'DR_Inf_Total_AUC': 'B3', 'DR_Inf_Total_1p0mgkg': 'B4',
                  'DR_Inf_Total_0p32mgkg': 'B5', 'DR_Inf_Total_0p1mgkg': 'B6',
                  'DR_Inf_Total_0p032mgkg': 'B7', 'EX_ALP_Total_s02': 'B8',
                  'RI_vs_Sal_ALP_Total_RIn_s01': 'B9',}, inplace=True)

mean_noncluster = IVSA_cluster_means.mean()
print(mean_noncluster)

#print(mean_noncluster[0])


# In[82]:


groups = [[], [], []]

for i in range(3):
    for index, item in enumerate(kmeans.labels_):
        if item == i:
            groups[i].append(index)

print(groups)


# In[83]:


cluster_1 = pd.DataFrame()
for i in groups[0]:
    cluster_1 = cluster_1.append(IVSA_cluster_means.iloc[i])
mean_cluster1 = cluster_1.mean()

graph1_values = []
for i in range(9):
    graph1_values.append(mean_cluster1[i])
    
cluster_1.head()


# In[84]:


cluster_2 = pd.DataFrame()
for i in groups[1]:
    cluster_2 = cluster_2.append(IVSA_cluster_means.iloc[i])
mean_cluster2 = cluster_2.mean()

graph2_values = []
for i in range(9):
    graph1_values.append(mean_cluster2[i])
    
cluster_2.shape


# In[85]:


cluster_3 = pd.DataFrame()
for i in groups[2]:
    cluster_3 = cluster_3.append(IVSA_cluster_means.iloc[i])
mean_cluster3 = cluster_3.mean()

graph3_values = []
for i in range(9):
    graph3_values.append(mean_cluster3[i])
    
cluster_3.shape


# In[86]:


fig, ax = plt.subplots(figsize=(10,8))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.set(ylim=(-3, 6))
sns.violinplot(data=cluster_1, width=1)


# In[87]:


fig, ax = plt.subplots(figsize=(10,8))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.set(ylim=(-3, 6))
sns.violinplot(data=cluster_2, width=1)


# In[88]:


fig, ax = plt.subplots(figsize=(10,8))
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)
ax.set(ylim=(-3, 6))
sns.violinplot(data=cluster_3, width = 1)


# In[60]:


from sklearn.decomposition import PCA
pca = PCA(n_components=4)

principalComponents = pca.fit_transform(behaviors)


# In[61]:


print(principalComponents)


# In[62]:


print(pca.explained_variance_ratio_.cumsum())


# In[63]:


print(pca.explained_variance_)


# In[64]:


cleaned_microbiome = pd.read_csv('non-normalized.csv')


# In[65]:


UMAP_kmeans = [str(values) for values in UMAP_kmeans]
behaviors["clusters"] = UMAP_kmeans


# In[66]:


behaviors


# In[67]:


behaviors.to_csv('behaviors.csv', index=False)


# In[68]:


cleaned_microbiome.info()


# In[69]:


df_cleaned = pd.read_csv("df_cleaned.csv")
df_cleaned = df_cleaned.iloc[:, 1:97]
UMAP_kmeans = [str(values) for values in UMAP_kmeans]
df_cleaned["clusters"] = UMAP_kmeans

df_cleaned.to_csv('df_cleaned.csv', index=False)


# In[ ]:




