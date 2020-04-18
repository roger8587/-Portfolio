#%% 
# =============================================================================
# 2.Support Vector Machine
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt
import matlab
import matlab.engine
import pandas as pd
x_train = pd.read_csv('x_train.csv',header = None)
x_train = np.array(x_train,dtype='float')
t_train = pd.read_csv('t_train.csv',header = None)
n_class = len(t_train[0].unique())
t_train = np.array(t_train,dtype='float')
def pca(df, topNfeat):
    df_mu = np.mean(df, axis=0)
    df_mat = df - df_mu.reshape(1,-1)           
    covdf = np.cov(df_mat, rowvar=0)
    eigVals,eigVects = np.linalg.eig(covdf) 
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:-(topNfeat+1):-1]  
    redEigVects = eigVects[:,eigValInd]       
    lowdf = df_mat @ redEigVects     
    return lowdf

x_train = pca(x_train,2)/225
x_train = x_train.real
#%%
def kernel_phi(x,k):
    if k == 0:
        return np.asarray(x).reshape(2,1)
    else: 
        return np.array([x[0]**2, np.sqrt(2)*x[0]*x[1], x[1]**2]).reshape(3,1)
kernel_phi(x_train[0],1)

def compute_kernel(data,k):
    n = len(data)
    k1 = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            k1[i][j] = kernel_phi(data[i],k).T.dot(kernel_phi(data[j],k))[0][0]
    return k1

#%%
  
def fit_each(data, y, k, c , threshold): 
    sv = np.zeros((1,2))
    n = len(data)
    a1 = compute_kernel(data,k)
    k_mat = matlab.double(a1.tolist())
    y_mat = matlab.double(y.tolist())
    y = y[0]
    eng = matlab.engine.start_matlab()
    alpha = np.asarray(eng.smo(k_mat ,y_mat ,c ,threshold)[0]).reshape(n)
    C = c/n
    Nm = 0 
    bias = 0.
    for i in range(n):
        if alpha[i] == 0. or alpha[i] == C:
            continue
        Nm += 1
        tmp = 0.
        for j in range(n): 
            if alpha[j] == 0.:
                continue
            tmp += alpha[j] * y[j] * a1[i][j]
            sv = np.vstack((sv,data[j]))
        bias += (y[i] - tmp)
    bias /= Nm
    w = np.zeros((kernel_phi(data[0],k).shape[0],1))
    for i in range(n):
        w += alpha[i] * y[i] * kernel_phi(data[i],k)
    return w, bias, sv[1:,:]
def fit(data, target, k, c, threshold):
    a2 = int(np.amax(target)) + 1 
    w_and_b = []
    sv1 = np.zeros((1,2))
    for s in range(a2):
        for t in range(a2):
            if s < t or s == t:
                continue
            tmp_data = []
            tmp_target = []
            for n in range(len(data)):
                if target[n] == s:
                    tmp_data += [data[n]]
                    tmp_target += [1]
                elif target[n] == t:
                    tmp_data += [data[n]]
                    tmp_target += [-1]
                else:
                    continue
            tmp_target = np.asarray(tmp_target).reshape(1,len(tmp_target))
            tmp_data = np.asarray(tmp_data)
            w, b, s_v = fit_each(tmp_data, tmp_target,k , c, threshold)
            w_and_b += [[w,b,s,t]]
            sv1 = np.vstack((sv1,s_v))
    return w_and_b, sv1[1:,:]
#%%
def predict(x,wb,k):
    if wb == -1:
        raise NameError('must fit dataset first')
    vote = [0] * n_class 
    for each in range(len(wb)):
        p = wb[each][0].T.dot(kernel_phi(x,k)) + wb[each][1]
        if p >= 0:
            vote[wb[each][2]] += 1
        else:
            vote[wb[each][3]] += 1
    return vote.index(max(vote))

def batch_predict(xx, yy,wb,k):
    if wb == -1:
        raise NameError('must fit dataset first')
    out = np.zeros(xx.shape, dtype = int)
    for r in range(xx.shape[0]):
        for c in range(xx.shape[1]):
            out[r][c] = predict([ xx[r][c], yy[r][c] ],wb,k)
    return out
def plot(train, target, k, c, threshold,title):
    wb,svv = fit(train, target, k, c, threshold)
    step = 0.01
    x_min, x_max = train[:, 0].min() - 1, train[:, 0].max() + 1
    y_min, y_max = train[:, 1].min() - 1, train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
    dist = batch_predict(xx,yy,wb,k)
    color = ['red', 'blue' ,'limegreen']
    plt.figure()
    plt.contourf(xx, yy, dist, alpha=0.3, levels=np.arange(dist.max()+2)-0.5, antialiased=True, colors = color)
    plt.scatter(svv[:,0], svv[:,1], color = 'black', marker = 'o',label='support vector')
    ax = []
    bx = []
    cx = []
    v = [ax, bx, cx]
    for i in range(len(train)):
        p = predict(train[i],wb,k)
        v[p] += [train[i]]
    ax = np.asarray(ax)
    bx = np.asarray(bx)
    cx = np.asarray(cx)
    plt.scatter(ax[:,0], ax[:,1], color = 'red', marker = 'x',label='class 0')
    plt.scatter(bx[:,0], bx[:,1], color = 'blue', marker = 'x',label='class 1')
    plt.scatter(cx[:,0], cx[:,1], color = 'limegreen', marker = 'x',label='class 2')
    plt.legend(loc='lower left', shadow=True)
    plt.title(title)
    plt.savefig(str(k)+'.png')

#%%
img_title = ['Linear Kernel','Polynomial Kernel']        
for i in range(2):
    plot(x_train ,t_train, i, 5., 0.,img_title[i])
    error = 0
    wb1 = fit(x_train, t_train, i, 5., 0.)[0]
    for n in range(len(x_train)):      
        if predict(x_train[n],wb1,i) != t_train[n]:
            error += 1
    print ('error:', error)