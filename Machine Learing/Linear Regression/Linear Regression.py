#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.core.frame import DataFrame

#2
# =============================================================================
# Feature selection
# =============================================================================
#(1-a)
def get_train_validation(labeled_df, validation_size=0.3):
  m = labeled_df.shape[0]
  validation_threshold = int(validation_size * m)
  validation = labeled_df.iloc[0:validation_threshold, :]
  train = labeled_df.iloc[validation_threshold:, :]
  return train, validation
train = pd.read_csv("dataset_X.csv")
target = pd.read_csv("dataset_T.csv")
data = pd.merge(train,target,how = 'left',on = '#Date')
train.drop(['#Date'],axis=1,inplace=True)  
target.drop(['#Date'],axis=1,inplace=True) 
train_df, validation_df = get_train_validation(train)
train_target, validation_target = get_train_validation(target)
print(train_df.shape)
print(validation_df.shape)

def get_thetas_lm(X, y):
  m = X.shape[0]
  y = y.reshape(-1, 1)
  ones = np.ones(m, dtype=int).reshape(-1, 1)
  X = np.concatenate([ones, X], axis=1)
  LHS = np.dot(X.T, X)
  RHS = np.dot(X.T, y)
  LHS_inv = np.linalg.inv(LHS)
  thetas = np.dot(LHS_inv, RHS)
  return thetas
X_train = train_df.values
y_train = train_target.values
weight = get_thetas_lm(X_train, y_train)

def rms(a,b,d):
    n = a.shape[0]
    ones = np.ones(n, dtype=int).reshape(-1, 1)
    X = np.concatenate([ones, a], axis=1)
    w_train = X.dot(d)
    return np.sqrt(np.sum(np.square(w_train.reshape(-1, 1)-b.reshape(-1, 1)))/n)

print(rms(X_train,y_train,weight),rms(validation_df.values,validation_target.values,weight))

# m = 2

X_train2=X_train
for i in range(17):
    for j in range(17-i):
        a=np.array([X_train[:,i]*X_train[:,j+i]])
        X_train2=np.append(X_train2,a.T,axis=1)
y_train2 = train_target.values
weight = get_thetas_lm(X_train2, y_train2)

print(rms(X_train2,y_train2,weight))
X_test2 = validation_df.values
for i in range(17):
    for j in range(17-i):
        a=np.array([validation_df.values[:,i]*validation_df.values[:,j+i]])
        X_test2=np.append(X_test2,a.T,axis=1)
print(rms(X_test2,validation_target.values,weight))

#(1-b)
a = list(train.columns)
d = []
for i in range(17):
    weight = get_thetas_lm(X_train[:,i].reshape(-1, 1), y_train)
    b = rms(X_train[:,i].reshape(-1, 1),y_train,weight)
    #print(a[i],'的 RMS =',b)
    d.append([a[i],b])
print(d)
#%%
# =============================================================================
# Maximum likelihood approach
# =============================================================================
#(2-a)
#feature selection

#維度4
a = list(train.columns)
f4 = []
for i in range(17):
    for j in range(i+1,17):
        for k in range(j+1,17):
            for m in range(k+1,17):
                weight = get_thetas_lm(X_train[:,[i,j,k,m]], y_train)
                b = rms(X_train[:,[i,j,k,m]],y_train,weight)
                f4.append([a[i],a[j],a[k],a[m],b])
f4 = DataFrame(f4)
print(f4[f4[4]<4.3])

#(2-b)
#N-fold cross-validation
def get_validation(labeled_df, range1, range2, validation_size=0.125):
  m = labeled_df.shape[0]
  validation_threshold = int(validation_size * m)
  validation = labeled_df.iloc[validation_threshold*range1:validation_threshold*range2, :]
  train = pd.concat([labeled_df.iloc[0:validation_threshold*range1, :],labeled_df.iloc[validation_threshold*range2:, :]],axis=0)
  return train, validation
train = pd.read_csv(r"C:\Users\517-B\Desktop\機器學習-簡仁宗\2019\[2019]ML_HW1\Dataset\dataset_X.csv")
target = pd.read_csv(r"C:\Users\517-B\Desktop\機器學習-簡仁宗\2019\[2019]ML_HW1\Dataset\dataset_T.csv")
data = pd.merge(train,target,how = 'left',on = '#Date')
train.drop(['#Date'],axis=1,inplace=True)  
target.drop(['#Date'],axis=1,inplace=True) 
train_df, validation_df = get_validation(train[['CO','PM10','NMHC','WIND_SPEED']],1,2)
train_target, validation_target = get_validation(target,1,2)
print(train_df.shape)
print(validation_df.shape)
print(train_target.shape)
print(validation_target.shape)

def model_selection(order,train):
    m = train.shape[0]
    model = np.ones(m, dtype=int).reshape(-1, 1)
    for i in range(order+1):
        for j in range(order+1-i):
            for k in range(order+1-i-j):
                for n in range(order+1-i-j-k):
                    a = (pow(train.values[:,0],i)*pow(train.values[:,1],j)*pow(train.values[:,2],k)*pow(train.values[:,3],n)).reshape(-1, 1)
                    model = np.append(model,a,axis=1)
    return(model[:,1:])
m1 = model_selection(3,train_df)
m2 = model_selection(3,validation_df)

def weight(model1,test):
    LHS = np.dot(model1.T, model1)
    RHS = np.dot(model1.T, test.values)
    LHS_inv = np.linalg.inv(LHS)
    w = np.dot(LHS_inv, RHS)
    return(w)
w1 = weight(m1,train_target)

def rm(model1,test,w):
    m = model1.shape[0]
    w_train1 = model1.dot(w)
    rms = np.sqrt(np.sum(np.square(w_train1.reshape(-1, 1)-test.values.reshape(-1, 1)))/m)
    return(rms)
print(rm(m1,train_target,w1))

model_rms = []
for j in range(1,8):
    rms_mean=[]
    rms_mean1=[]
    for i in range(8):
        train_df, validation_df = get_validation(train[['CO','PM10','NMHC','WIND_SPEED']],i,i+1)
        train_target, validation_target = get_validation(target,i,i+1)
        m1 = model_selection(j,train_df)
        m2 = model_selection(j,validation_df)
        w1 = weight(m1,train_target)
        rms1 = rm(m1,train_target,w1)
        rms = rm(m2,validation_target,w1)
        rms_mean.append(rms)
        rms_mean1.append(rms1)
    print(j,np.mean(rms_mean1),np.mean(rms_mean))
    model_rms.append([np.mean(rms_mean1),np.mean(rms_mean)])
model_rms = DataFrame(model_rms)

m = range(1,6)
plt.plot(m,model_rms[0][0:5])
plt.plot(m,model_rms[1][0:5])
plt.xlabel('Order')
plt.ylabel('Cross-Validated RMS')
plt.show()
#%%
# =============================================================================
# Maximum a posteriori approach
# =============================================================================
def weight2(model1,test):
    LHS = np.dot(model1.T, model1)+np.eye(m1.shape[1])*0.15
    RHS = np.dot(model1.T, test.values)
    LHS_inv = np.linalg.inv(LHS)
    w = np.dot(LHS_inv, RHS)
    return(w)
w2 = weight2(m1,train_target)
print(rm(m1,train_target,w2))

model_rms1 = []
for j in range(1,8):
    rms_mean=[]
    rms_mean1=[]
    for i in range(8):
        train_df, validation_df = get_validation(train[['CO','PM10','NMHC','WIND_SPEED']],i,i+1)
        train_target, validation_target = get_validation(target,i,i+1)
        m1 = model_selection(j,train_df)
        m2 = model_selection(j,validation_df)
        w1 = weight2(m1,train_target)
        rms1 = rm(m1,train_target,w1)
        rms = rm(m2,validation_target,w1)
        rms_mean.append(rms)
        rms_mean1.append(rms1)
    print(j,np.mean(rms_mean1),np.mean(rms_mean))
    model_rms1.append([np.mean(rms_mean1),np.mean(rms_mean)])
model_rms1 = DataFrame(model_rms1)
    
m = range(1,6)
plt.plot(m,model_rms1[0][0:5],label='train')
plt.plot(m,model_rms1[1][0:5],label='test')
plt.xlabel('Order')
plt.ylabel('Cross-Validated RMS')
plt.show()






