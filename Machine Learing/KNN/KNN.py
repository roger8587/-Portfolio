#%%
#前處理
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('Pokemon.csv')
df = df.drop(["Unnamed: 0","Name"], axis = 1)

title_mapping = {False: 0, True: 1}
df['Legendary'] = df['Legendary'].map(title_mapping)
title_mapping2 = {'Normal': 1, 'Water': 2,'Psychic':3}
df['Type 1'] = df['Type 1'].map(title_mapping2)

for i in range(8):
    df.iloc[:,i+1] = (df.iloc[:,i+2]-df.iloc[:,i+2].mean())/df.iloc[:,i+2].std()
x_train = df.iloc[:120,:]
x_test = df.iloc[120:,:]

#%%
#1. K-nearest-neighbor classi

def euclideanDistance(data, test, n1, n2): 
    d1=np.sqrt(np.sum(np.square(data.iloc[n1,1:]-test.iloc[n2,1:])))
    return(d1) 

def pred(data,test):
    eudis = []
    predict = []
    true = []
    for i in range(test.shape[0]):
        dis = []
        p = []
        for j in range(data.shape[0]):
            dis.append(euclideanDistance(data, test, j, i))
            p.append(data.iloc[j,0])
        eudis.append(dis)
        predict.append(p)
        true.append(test.iloc[i,0])
    return(eudis,predict,true)
    
eudis,predict,true = pred(x_train,x_test)   
    
def select_k(dis,p,k):
    pred=[]
    for i in range(len(dis)): 
        idx = np.argsort(dis[i])
        pred.append(np.array(p[i])[idx][:k])
    pred = np.array(pred)
    if k==1:
        return(np.array(pred))
    else:
        pp = []
        for j in range(len(pred)):
            d = {x:list(pred[j]).count(x) for x in list(pred[j])}
            a,b = list(d.keys()),list(d.values())
            pp.append(a[b.index(max(b))])
        return(np.array(pp))    
def accuracy(p,t):
    h = 0
    for i in range(len(p)):
        if p[i] == np.array(t)[i]:
            h += 1
    return(h/len(p))    
    
acc = []
for k in range(1,11):
    acc.append(accuracy(select_k(eudis,predict,k),true))
acc        
k = [i for i in range(1,11)]    
plt.plot(k,acc)
plt.xlabel('k')
plt.ylabel('accuracy')    
plt.title('KNN')    
plt.savefig('KNN.png')    
#%%   
#(2).PCA 
def pca(df,k):
    xtx = np.dot(df.T,df)
    [df_eig_v,df_eig_vec] = np.linalg.eigh(xtx)
    df_eig_vec = np.dot(df,df_eig_vec)
    for i in range(df.shape[0]):
        df_eig_vec[i] = df_eig_vec[i]/np.linalg.norm(df_eig_vec[i])
    idx = np.argsort(-df_eig_v)
    df_eig_v = df_eig_v[idx]
    df_eig_vec = df_eig_vec[:,idx]
    df_eig_v = df_eig_v[0:k].copy()
    df_eig_vec = df_eig_vec[:,0:k].copy()
    return(df_eig_vec) 
    
df1 = df.iloc[:,1:]
m = [7,6,5]
for n in m:
    x2_train ,x2_test = pca(df1,n)[:120,:] , pca(df1,n)[120:,:]
    y1 = np.array(df.iloc[:120,0]).reshape(1,-1)
    y2 = np.array(df.iloc[120:,0]).reshape(1,-1)
    x2_train = np.insert(x2_train, 0, y1, axis=1)
    x2_test = np.insert(x2_test, 0, y2, axis=1)
    x2_train = pd.DataFrame(x2_train) 
    x2_test = pd.DataFrame(x2_test)
    eudis,predict,true = pred(x2_train,x2_test)
    acc = []
    for k in range(1,11):
        acc.append(accuracy(select_k(eudis,predict,k),true))
    k1 = [i for i in range(1,11)]
    plt.plot(k1,acc)
    plt.xlabel('k')
    plt.ylabel('accuracy')
    plt.title('KNN(M='+str(n)+')')
    plt.savefig('KNN(M='+str(n)+').png')
    plt.show()    