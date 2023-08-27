#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install scikit-plot


# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_squared_error
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from torch.autograd import Variable
from scipy.stats import ranksums
import random


# In[5]:


percentage_identification = pd.read_csv('non-normalized.csv')
clusters = pd.read_csv('behaviors.csv')

percentage_identification = pd.concat([percentage_identification, clusters], axis=1)
percentage_identification = percentage_identification.drop(['AQ_SessionsToAcquisition', 'DR_StS_1p0mgkg', 'DR_Inf_Total_AUC',
                                         'DR_Inf_Total_1p0mgkg', 'DR_Inf_Total_0p32mgkg', 'DR_Inf_Total_0p1mgkg', 
                                         'DR_Inf_Total_0p032mgkg', 'EX_ALP_Total_s02', 'RI_vs_Sal_ALP_Total_RIn_s01'], axis=1)


# In[7]:


from scipy.spatial import distance
import seaborn as sns
import statistics

to_analyze = pd.read_csv('df_cleaned.csv').values.tolist()
braycurtis_arr = pd.read_csv('non-normalized.csv').values.tolist()

clster0 = []
clster1 = []
clster2 = []

for i in range(len(to_analyze)):
    if to_analyze[i][-1] == 0:
        clster0.append(braycurtis_arr[i])
    elif to_analyze[i][-1] == 1:
        clster1.append(braycurtis_arr[i])
    else:
        clster2.append(braycurtis_arr[i])
        
val = clster0 + clster1 + clster2

braycurtis_values = []
for i in range(len(val)):
    lst = []
    for j in range(len(val)):
        lst.append(distance.braycurtis(val[i], val[j]))
    braycurtis_values.append(lst)

    
plt.figure(figsize=(10,8))
braycurtis_values = np.array(braycurtis_values)
heat_map = sns.heatmap(braycurtis_values)
plt.title( "HeatMap for Microbiome Beta-Diversity", size=20)
plt.show()

denom = 0
numer = 0
for i in braycurtis_values:
    for a in i:
        denom += 1
        if a <= 0.35 and a > 0.0:
            numer += 1
            
print(numer/denom)

total = 0
for i in braycurtis_values:
    total += statistics.mean(i)


# In[70]:


csv_file = pd.read_csv('normalized.csv')
clusters = pd.read_csv('df_cleaned.csv')

csv_file = pd.concat([csv_file, clusters["clusters"]], axis=1)

csv_file.head()


# In[71]:


X = csv_file.iloc[:, :-1]
y = csv_file.iloc[:, -1]

skf = StratifiedKFold(n_splits=5, shuffle = True, random_state=80) #20, 0, 80
skf.get_n_splits(X, y)

for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "\n\n", "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    break
        
trainset = pd.concat([X_train, y_train], axis=1)
testset = pd.concat([X_test, y_test], axis=1)

trainset = trainset.reset_index()
testset = testset.reset_index()

trainset = trainset.drop('index', axis=1)
testset = testset.drop('index', axis=1)

#print(X_test.shape)

a = trainset.index.tolist()
b = testset.index.tolist()

#print(X_train.columns.tolist)
#print(X_test.columns.tolist)

lst =[]
for i in a:
    lst.append(csv_file.iloc[i]['clusters'])
    
print(lst.count(0))
print(lst.count(1))
print(lst.count(2))

print("\n")

lst2 =[]
for i in b:
    lst2.append(csv_file.iloc[i]['clusters'])
    
print(lst2.count(0))
print(lst2.count(1))
print(lst2.count(2))


# In[72]:


trainset.to_csv('trainset.csv', index=False)
testset.to_csv('testset.csv', index=False)


# In[73]:


#  ---------------  Dataset  ---------------
class MicrobiomeClusterDataset(Dataset):
    """Students Performance dataset."""

    def __init__(self, csv_file):
        """Initializes instance of class StudentsPerformanceDataset.
        Args:
            csv_file (str): Path to the csv file with the students data.
        """
        df = csv_file
        
        col_names = list(df.columns.values)

        # Grouping variable names
        self.categorical = col_names[0:-1]
        self.target = col_names[-1]

        # One-hot encoding of categorical variables
        self.microbiomes_frames = df

        # Save target and predictors
        self.X = df.drop(self.target, axis=1)
        self.y = df[self.target]

    def __len__(self):
        return len(self.microbiomes_frames)

    def __getitem__(self, idx):
        # Convert idx from tensor to list due to pandas bug (that arises when using pytorch's random_split)
        # Can apply random augmentation function here, to allow on-the-fly data augmentation. 
        if isinstance(idx, torch.Tensor):
            idx = idx.tolist()

        return [self.X.iloc[idx].values, self.y[idx]]


# In[74]:


#  ---------------  Model  ---------------

class Net(nn.Module):
    #classification D_out = number of clusters
    def __init__(self, D_in, H=15, D_out=3):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, D_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)

        return x


# In[75]:


def weight_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)


# In[76]:


#  ---------------  Model  ---------------

class Net2(nn.Module):
    #classification D_out = number of clusters
    def __init__(self, D_in, H1=15, H2 = 3, D_out=3):
        super().__init__()
        self.fc1 = nn.Linear(D_in, H1)
        self.fc2 = nn.Linear(H1, H2)
        self.fc3 = nn.Linear(H2, D_out)
        self.relu = nn.ReLU()
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)

        return x


# In[77]:


# Load dataset
trainset = MicrobiomeClusterDataset(trainset)
testset = MicrobiomeClusterDataset(testset)

print(trainset.y)


# In[78]:


# Dataloaders
trainloader = DataLoader(trainset, batch_size=60, shuffle=True)
testloader = DataLoader(testset, batch_size=200, shuffle=False)


# In[79]:


#  ---------------  Training  ---------------
n_epochs = 500

print(trainset)

# Use gpu if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define the model
D_in, H = csv_file.shape[1]-1, 15 # D_in is realated to input data, H (hidden dimension) is a hyper-parameter that can be tuned.
#model = Net(D_in, H=3)
model = Net2(D_in, H1=20, H2=5).to(device)
torch.manual_seed(1) #1
model.apply(weight_init)

# Loss function
criterion = nn.CrossEntropyLoss()
# Optimiz
optimizer = optim.Adam(model.parameters(), weight_decay=0.01)

# Train the net
loss_per_iter = []
loss_per_batch = []
for epoch in range(n_epochs):

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward + backward + optimize
        outputs = model(inputs.float())
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

        # Save loss to plot
        running_loss += loss.item()
        loss_per_iter.append(loss.item())

    loss_per_batch.append(running_loss / (i + 1))
    running_loss = 0.0

# Plot training loss curve
plt.plot(np.arange(len(loss_per_iter)), loss_per_iter, "-", alpha=0.5, label="Loss per mini-batch")
plt.xlabel("Number of iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()


# In[80]:


def single_class_auc(idx, y_true, y_pred):
    label_flag = y_true == idx
    pred_flag = y_pred[:,idx]
    roc_auc = roc_auc_score(label_flag, pred_flag)
    return roc_auc

def test_classification_model2(model, dataloader):
   
    all_preds = np.zeros((0, 3))
    all_labels = np.zeros((0))
   
    
    model.eval()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float())
       
        all_labels = np.append(all_labels, labels.data.numpy(), axis=0)
        all_preds = np.append(all_preds, outputs.data.numpy(), axis=0)
       
    all_preds = softmax(all_preds, axis=1)
   
    for idx in range(3):
        print(single_class_auc(idx, all_labels, all_preds))
    roc_auc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')
   
    return roc_auc

print("Trainset ROC_AUC: ", test_classification_model2(model, trainloader))
print("Testset ROC_AUC: ", test_classification_model2(model, testloader))


# In[93]:


def test_classification_model(model, dataloader):
   
    all_preds = np.zeros((0, 3))
    all_labels = np.zeros((0))
    original_data = np.zeros((0, 69)) #originally 57
   
    
    model.eval()
    for i, (inputs, labels) in enumerate(dataloader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs.float())

        #print("Labels = {}, Outputs = {}".format(labels.data.numpy(), softmax(outputs.data.numpy(), axis=1)))
       
        all_labels = np.append(all_labels, labels.data.numpy(), axis=0)
        all_preds = np.append(all_preds, outputs.data.numpy(), axis=0)
        original_data = np.append(original_data, inputs.data.numpy(), axis=0)
       
    all_preds = softmax(all_preds, axis=1)
   
    for idx in range(3):
        print(single_class_auc(idx, all_labels, all_preds))
    roc_auc = roc_auc_score(all_labels, all_preds, average='weighted', multi_class='ovr')
   
    return roc_auc, all_labels, all_preds, original_data


# In[94]:


train_auc, train_label, train_pred, train_input = test_classification_model(model, trainloader)
test_auc, test_label, test_pred, test_input = test_classification_model(model, testloader)

#print(train_auc, test_auc)

#print(train_label)
#print(trainset.y.tolist())

correct_predict_train = []

for i, item in enumerate(train_pred):
    if item.argmax() == train_label[i]:
        correct_predict_train.append(i)

train_correct = train_input[correct_predict_train] #microbiome data
train_label_correct = train_label[correct_predict_train] #cluster labels

print(train_correct.shape)
print(train_label_correct.shape)

train_filter = np.concatenate((train_correct, train_label_correct.reshape(-1, 1)), axis=1) #train_correct and train_label... concatenated

#numpy.savetxt("microb_for_trainloader.csv", )
microb_for_trainloader = pd.DataFrame(train_filter, columns = csv_file.columns)

trainset2 = MicrobiomeClusterDataset(microb_for_trainloader)
trainloader2 = DataLoader(trainset2,  batch_size=1)


# In[95]:


print(test_pred)
print(test_label)


# In[96]:


import scikitplot as skplt
import matplotlib.pyplot as plt

y_true_for_curve = []
for i in test_label:
    if i == 0:
        y_true_for_curve.append('High Risk')
    if i == 1:
        y_true_for_curve.append('Low Use')
    if i == 2:
        y_true_for_curve.append('Low Risk')
        
y_true_for_curve

y_true = [int(x)+1 for x in test_label]  #y_true_for_curve
y_probas = test_pred

fig, ax1 = plt.subplots(figsize=(10, 7))
disp = skplt.metrics.plot_roc_curve(y_true, y_probas, ax=ax1, text_fontsize=15)

plt.show()


# In[97]:


print(microb_for_trainloader.head())

print(trainset2[0])


# In[20]:


indices_0 = []
indices_1 = []
indices_2 = []


# In[21]:


trainloader = DataLoader(trainset, batch_size=1, shuffle=True)
List_CAMs_0 = []
List_CAMs_1 = []
List_CAMs_2 = []

for i, (inputs, labels) in enumerate(trainloader2):
    inputs = inputs.to(device)
    inputs = Variable(inputs, requires_grad=True)
    labels = labels.to(device)
    
    #print(inputs.shape)
    #print(labels.shape)
    
    labels = labels.reshape(-1, 1)

    # Zero the parameter gradients
    optimizer.zero_grad()

    # Forward + backward + optimize
    outputs = model(inputs.float())
    #print(outputs)
    #print(labels.long())
    o2 = (outputs * ((labels.long()-0.5)*2))[0][0]
    o2.backward()
    CAM_Current = (inputs.data * inputs.grad)[0]
    if labels.long()[0].item()==0:
        List_CAMs_0.append(CAM_Current)
        indices_0.append(i)
    elif labels.long()[0].item()==1:
        List_CAMs_1.append(CAM_Current)
        indices_1.append(i)
    else:
        List_CAMs_2.append(CAM_Current)
        indices_2.append(i)
List_CAMs_Combined01 = List_CAMs_0 + List_CAMs_1
List_CAMs_Combined02 = List_CAMs_0 + List_CAMs_2
List_CAMs_Combined12 = List_CAMs_1 + List_CAMs_2

print(List_CAMs_0)


# In[22]:


x = [micro for micro in microb_for_trainloader.columns]
graph_length = len(x) - 1


# In[37]:


List_Statistic1 = []
List_PValues1 = []
for i in range(graph_length-1):
    array0 = np.array([a[i].item() for a in List_CAMs_0])
    array1 = np.array([a[i].item() for a in List_CAMs_Combined12])
    statistic, pvalue = ranksums(array0,array1)
    pvalue = pvalue * 57
    List_Statistic1.append(statistic) 
    List_PValues1.append(pvalue) #multiple by number of microbiome features in order to do Boferonni Test
    print(pvalue)


# In[39]:


List_Statistic2 = []
List_PValues2 = []
for i in range(graph_length-1):
    array0 = np.array([a[i].item() for a in List_CAMs_1]) 
    array1 = np.array([a[i].item() for a in List_CAMs_Combined02]) 
    statistic, pvalue = ranksums(array0,array1)
    pvalue = pvalue * 57
    List_Statistic2.append(statistic)
    List_PValues2.append(pvalue) #multiple by number of microbiome features in order to do Boferonni Test
    print(pvalue)


# In[40]:


List_Statistic3 = []
List_PValues3 = []
for i in range(graph_length-1):
    array0 = np.array([a[i].item() for a in List_CAMs_2]) #Low_Use
    array1 = np.array([a[i].item() for a in List_CAMs_Combined01]) #Low_Risk
    statistic, pvalue = ranksums(array0,array1)
    pvalue = pvalue * 57
    List_Statistic3.append(statistic)
    List_PValues3.append(pvalue) #multiple by number of microbiome features in order to do Boferonni Test


# In[41]:


import matplotlib.pyplot as plt


# In[42]:


Select_BestN = 4
fig, ax = plt.subplots(figsize=(4,4))
plt.yticks(fontsize=18)
plt.xticks(fontsize=16)
plt.xlabel("-Log P", fontsize=18)

Array_Current = -np.log10(np.array(List_PValues1))
Index_Sorted = Array_Current.argsort()
Y_Names = [micro for micro in microb_for_trainloader.columns]

plt.barh(range(Select_BestN), Array_Current[Index_Sorted][-Select_BestN:], 0.5)
plt.yticks(range(Select_BestN),[Y_Names[Index_Sorted[-Select_BestN+i]] for i,a in enumerate(range(Select_BestN))])


# In[43]:


fig, ax = plt.subplots(figsize=(4,4))
plt.yticks(fontsize=18)
plt.xticks(fontsize=16)
plt.xlabel("-Log P", fontsize=18)

Array_Current = -np.log10(np.array(List_PValues2))
Index_Sorted = Array_Current.argsort()
Y_Names = [micro for micro in microb_for_trainloader.columns]

plt.barh(range(Select_BestN), Array_Current[Index_Sorted][-Select_BestN:], 0.5)
plt.yticks(range(Select_BestN),[Y_Names[Index_Sorted[-Select_BestN+i]] for i,a in enumerate(range(Select_BestN))])


# In[44]:


fig, ax = plt.subplots(figsize=(4,4))
plt.yticks(fontsize=18)
plt.xticks(fontsize=16)
plt.xlabel("-Log P", fontsize=18)

Array_Current = -np.log10(np.array(List_PValues3))
Index_Sorted = Array_Current.argsort()
Y_Names = [micro for micro in microb_for_trainloader.columns]

plt.barh(range(Select_BestN), Array_Current[Index_Sorted][-Select_BestN:], 0.5)
plt.yticks(range(Select_BestN),[Y_Names[Index_Sorted[-Select_BestN+i]] for i,a in enumerate(range(Select_BestN))])


# In[30]:


csv_file.columns.tolist().index("Escherichia_Shigella")
csv_file["Escherichia_Shigella"]


# In[31]:


zeros = microb_for_trainloader.index[(microb_for_trainloader['clusters']==0)].tolist()
to_analyze0 = pd.DataFrame()

for i in range(len(zeros)):
    to_analyze0 = to_analyze0.append(microb_for_trainloader.iloc[zeros[i],:])
    
ones = microb_for_trainloader.index[(microb_for_trainloader['clusters']==1)].tolist()
to_analyze1 = pd.DataFrame()

for i in range(len(ones)):
    to_analyze1 = to_analyze1.append(microb_for_trainloader.iloc[ones[i],:])

twos = microb_for_trainloader.index[(microb_for_trainloader['clusters']==2)].tolist()
to_analyze2 = pd.DataFrame()

for i in range(len(twos)):
    to_analyze2 = to_analyze1.append(microb_for_trainloader.iloc[twos[i],:])
    
prominent_values = ['Enterococcus', 'Ruminococcus', 'Escherichia_Shigella', 'Akkermansia']

data_interpretation_dict = dict()

for i in prominent_values:
    data_interpretation_dict[i] = (to_analyze0[i].mean(), to_analyze1[i].mean(), to_analyze2[i].mean())
    
print(data_interpretation_dict)


# In[32]:


zeros = microb_for_trainloader.index[(microb_for_trainloader['clusters']==0)].tolist()
to_analyze0 = pd.DataFrame()

for i in range(len(zeros)):
    to_analyze0 = to_analyze0.append(microb_for_trainloader.iloc[zeros[i],:])
    
ones = microb_for_trainloader.index[(microb_for_trainloader['clusters']==1)].tolist()
to_analyze1 = pd.DataFrame()

for i in range(len(ones)):
    to_analyze1 = to_analyze1.append(microb_for_trainloader.iloc[ones[i],:])

twos = microb_for_trainloader.index[(microb_for_trainloader['clusters']==2)].tolist()
to_analyze2 = pd.DataFrame()

for i in range(len(twos)):
    to_analyze2 = to_analyze1.append(microb_for_trainloader.iloc[twos[i],:])
    
prominent_values = ['Escherichia_Shigella', 'Erysipelotrichaceae_incertae_sedis', 'ClostridiumXVIII', 'Anaerotruncus']

data_interpretation_dict = dict()

for i in prominent_values:
    data_interpretation_dict[i] = (to_analyze0[i].mean(), to_analyze1[i].mean(), to_analyze2[i].mean())
    
print(data_interpretation_dict)


# In[33]:


zeros = microb_for_trainloader.index[(microb_for_trainloader['clusters']==0)].tolist()
to_analyze0 = pd.DataFrame()

for i in range(len(zeros)):
    to_analyze0 = to_analyze0.append(microb_for_trainloader.iloc[zeros[i],:])
    
ones = microb_for_trainloader.index[(microb_for_trainloader['clusters']==1)].tolist()
to_analyze1 = pd.DataFrame()

for i in range(len(ones)):
    to_analyze1 = to_analyze1.append(microb_for_trainloader.iloc[ones[i],:])

twos = microb_for_trainloader.index[(microb_for_trainloader['clusters']==2)].tolist()
to_analyze2 = pd.DataFrame()

for i in range(len(twos)):
    to_analyze2 = to_analyze1.append(microb_for_trainloader.iloc[twos[i],:])
    
prominent_values = ['Fusicatenibacter', 'ClostridiumXVIII', 'Erysipelotrichaceae_incertae_sedis', 
                    'Parvibacter']

data_interpretation_dict = dict()

for i in prominent_values:
    data_interpretation_dict[i] = (to_analyze0[i].mean(), to_analyze1[i].mean(), to_analyze2[i].mean())
    
print(data_interpretation_dict)


# In[34]:


to_analyze0.shape


# In[94]:


to_analyze1.shape


# In[95]:


to_analyze2.shape


# In[96]:


print(len(microb_for_trainloader.index[(microb_for_trainloader['clusters']==0)].tolist()))
print(len(microb_for_trainloader.index[(microb_for_trainloader['clusters']==1)].tolist()))
print(len(microb_for_trainloader.index[(microb_for_trainloader['clusters']==2)].tolist()))


# In[ ]:




