# =============================================================================
# 2. Logistic Regression
# =============================================================================
#%% 
    
from PIL import Image    
import os
import numpy as np
import random
import matplotlib.pyplot as plt
plt.rcParams["figure.figsize"] = (5, 3) # (w, h)
plt.rcParams["figure.dpi"] = 200
im1 = []
classlabel = []
class1=0
for dirname, dirnames, filenames in os.walk(r'C:\Users\517-B\Desktop\Faces'):
    for subdirname in dirnames:
        sub_path = os.path.join(dirname, subdirname)
        for filename in os.listdir(sub_path):
            im = Image.open(os.path.join(sub_path,filename))
            im = im.convert('L')
            im1.append(np.asarray(im, dtype = np.uint8))
            classlabel.append(class1)
        class1 += 1
    
 #%% 
random.seed(124)
a=[random.sample(range(0,10), 5) for i in range(5) ]
b=list(range(10))
list(set(b)-set(a[1]))   
    
random.seed(124)
a=np.array([random.sample(range(0,10), 5) for i in range(5) ])
a1=list(range(10))
train_data = []
test_data = []
train_t = []
test_t = []
b = 0
for i in a:
    for j in range(50):
        if j in i+b:
            train_data.append(im1[j].reshape(1,-1))
            train_t.append(classlabel[j])
        elif j in np.array(list(set(a1)-set(i)))+b:
            test_data.append(im1[j].reshape(1,-1)) 
            test_t.append(classlabel[j])
    b+=10
train_data ,test_data , train_t , test_t= np.array(train_data).reshape(-1,10304) ,np.array(test_data).reshape(-1,10304) ,np.array(train_t) ,np.array(test_t)
print(len(test_t),len(test_data),len(train_t),len(train_data))
df = np.insert(train_data, 0, train_t, axis=1)
df1 = np.insert(test_data, 0, test_t, axis=1)
np.random.seed(135) 
np.random.shuffle(df)
np.random.shuffle(df1)
x_train, y_train = df[:,1:], df[:,0]
x_test, y_test = df1[:,1:], df1[:,0]
tt = []
for i in y_train:
    t = [0 for k in range(5)]
    t[i] = 1
    tt.append(t)
y_train = np.array(tt)
t1 = []
for i in y_test:
    t = [0 for k in range(5)]
    t[i] = 1
    t1.append(t)
y_test = np.array(t1)
df1 = np.vstack((x_train,x_test))
#%%
def norm(dataMat):
    average = np.mean(dataMat, axis=1) #按列求均值
    m, n = np.shape(dataMat)
    meanRemoved = train_data-average.reshape(-1,1) #减去均值
    normData = meanRemoved / np.std(dataMat, axis=1).reshape(-1,1) #标准差归一化
    
    return normData
x_train, x_test = norm(x_train), norm(x_test)
#%%

def phi(x):
    x = np.reshape(x,(len(x),1))
    return x

    
def y(n,k,w,X): #softmax
    s = np.float64(0.)
    ak = w[k,:].T.dot(phi(X[n,:]))
    for j in range(5):
        aj = w[j,:].T.dot(phi(X[n,:]))
        s += np.nan_to_num(np.exp(aj - ak))
    s = np.nan_to_num(s)
    return 1./s
       
def gradient(w,k,t,X):
	output = np.zeros((len(w[0]),1))
	for n in range(len(X)):
		scale = y(n,k,w,X) - t[:,k][n] #Ynk - Tnk
		output += scale * phi(X[n])
	return output
    
def hessian(w,k,X):
	output = np.zeros((len(w[0]),len(w[0])))
	for n in range(len(X)): 
		scale = y(n,k,w,X) * (1 - y(n,k,w,X))
		output += scale * (phi(X[n]).dot(phi(X[n]).T)) 
	return output
    
def error(w,t,x):
	s = 0
	for n in range(x.shape[0]):
		for k in range(5):
			s += t[n,k]*np.nan_to_num(np.log(y(n,k,w,x)))
	return -1*s

def classify(x,w): 
	sfm = []
	for k in range(5):
		tt = 0
		ak = w[k].T.dot(phi(x))
		for k in range(5):
			aj = w[k,:].T.dot(phi(x))
			tt =tt+ np.nan_to_num(np.e**(aj - ak))
		sfm.append(1/tt)
	return sfm.index(max(sfm))
def accuracy(x,t,w): 
    ac=0
    for i in range(x.shape[0]):
        if t[i,classify(x[i,:],w)] ==1:
            ac=ac+1
    return ac/x.shape[0]
#%%

w = np.zeros((5,len(phi(x_train[0,:])),1))

cee = []
acc = []
lr = 6
while True:
	e = error(w,y_train, x_train)[0,0]
	acc.append([accuracy(x_train,y_train,w)])
	cee.append([np.reshape(e,1)])
	if e < 0.001:
		break
	for k in range(5):
		w[k,:] = w[k,:] - lr*(gradient(w,k,y_train,x_train))
w1 = w
epoch2 = [i for i in range(len(cee))]
plt.plot(epoch2, np.array(cee).reshape(-1,1))
plt.xlabel('epoch')
plt.ylabel('cross entropy error')
plt.xticks(epoch2)
plt.title('leanrinig curve (train)')
plt.savefig('2-1 learning_curve.png')
plt.show()

plt.plot(epoch2, acc)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.xticks(epoch2)
plt.title('accuracy curve (train)')
plt.savefig('2-1 accuracy curve.png')
plt.show()

#%%
#2-2

def confusion_matrix(pred,true):
    matrix = np.zeros((5,5))
    for i in range(len(pred)):
        matrix[true[i],pred[i]] += 1
    return(matrix)
x_value=[]
y_value=[]
for i in range(x_test.shape[0]):
    x_value.append(classify(x_test[i,:],w1))
    y_value.append(list(y_test[i,:]).index(1))
confusion_matrix(x_value,y_value)


#%%
#2-3
def pca(df,k):
    [n,d] = df.shape    
    df_mu = np.mean(df, axis=1)
    df_mat = (df1 - df_mu.reshape(-1,1))/np.std(df1, axis=1).reshape(-1,1)
    if n>d:
        xtx = np.dot(df_mat.T,df_mat)
        [df_eig_v,df_eig_vec] = np.linalg.eigh(xtx)
    else:
        xtx = np.dot(df_mat,df_mat.T)
        [df_eig_v,df_eig_vec] = np.linalg.eigh(xtx)
    df_eig_vec = np.dot(df_mat.T,df_eig_vec)
    for i in range(n):
        df_eig_vec[:,i] = df_eig_vec[:,i]/np.linalg.norm(df_eig_vec[:,i])
    idx = np.argsort(-df_eig_v)
    df_eig_v = df_eig_v[idx]
    df_eig_vec = df_eig_vec[:,idx]
    df_eig_v = df_eig_v[0:k].copy()
    df_eig_vec = df_eig_vec[:,0:k].copy()
    return(df_mat.dot(df_eig_vec))

pca(df1,5)
x2_train = pca(df1,5)[:25,:]
x2_test = pca(df1,5)[25:,:]
#%%
#2-4
M = [2,5,10]
for i in M: 
    print('--------------------------M=',i,'------------------------------')
    x2_train = pca(df1,i)[:25,:]
    x2_test = pca(df1,i)[25:,:]
    w2 = np.zeros((5,len(phi(x2_train[0,:])),1))
    cee2 = []
    acc2 = []
    
    for m in range(20):       
        e2_R = error(w2,y_train,x2_train)[0,0]
        acc2.append(accuracy(x2_train,y_train,w2))
        cee2.append(e2_R)
        for k in range(5):
            w2[k,:] = w2[k,:] - np.linalg.inv(hessian(w2,k,x2_train)).dot(gradient(w2,k,y_train,x2_train))

    fw2=w2
 
    epoch2_R = [a for a in range(len(cee2))]
    plt.plot(epoch2_R, cee2)
    plt.xlabel('epoch')
    plt.ylabel('cross entropy error')
    #plt.xticks(epoch2_R)
    plt.title('leanrinig curve (Netwon-Raphson) M='+str(i))
    plt.savefig('cross_entropy_error M='+str(i)+'.png')
    plt.show()
    
    plt.plot(epoch2_R, acc2)
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.xticks(epoch2_R)
    plt.title('accuracy curve (Netwon-Raphson) M='+str(i))
    plt.savefig('accuracy curve M='+str(i)+'.png')
    plt.show()
    
    x2_value=[]
    y2_value=[]
    for j in range(x2_test.shape[0]):
        x2_value.append(classify(x2_test[j,:],fw2))
        y2_value.append(list(y_test[j,:]).index(1))

    print(confusion_matrix(x2_value,y2_value))