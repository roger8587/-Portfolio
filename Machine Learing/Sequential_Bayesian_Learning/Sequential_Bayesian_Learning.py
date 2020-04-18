import scipy.io as sio 
import matplotlib.pyplot as plt 
import numpy as np 
from numpy.random import multivariate_normal as normal
import scipy.stats as stats
# =============================================================================
# 1.Sequential Bayesian Learning
# =============================================================================
#%%
data=sio.loadmat('1_data.mat')
x = data['x']
t = data['t']
j = np.arange(0,3,1)
mu = j*2/3
beta = 1
alpha = 10**-6
N = [5,10,30,80]
s = 0.1
a = []
for i in range(3):
    a.append((x - mu[i])/0.1)
#%%
def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
a1 = np.concatenate((sigmoid(a[0]),sigmoid(a[1]),sigmoid(a[2])),axis=1)
#%%
for n in N: 
#   1-1 
    plt.figure()
    plt.plot(x[:n],t[:n],'o')
    target = t[:n]
    PHI = a1[:n]
    s0_inv = (10**-6) * np.identity(3)
    sn_inv = s0_inv + PHI.T.dot(PHI)
    sn = np.linalg.inv(sn_inv)
    mn = sn.dot(PHI.T.dot(target))
    mn_ = np.reshape(mn, 3)
    line = np.linspace(0., 2., 50)
    ws = normal(mn_, sn, 5)
    for w in ws:
        line1 = []
        for i in range(3):
            line1.append((line - mu[i])/0.1)
        line1 = np.concatenate((sigmoid(line1[0]).reshape(50,1),sigmoid(line1[1]).reshape(50,1),sigmoid(line1[2]).reshape(50,1)),axis=1)
        value = []
        for point in line1:
            value += [point.T.dot(w)]
        plt.plot(line, value, linestyle ='--', zorder = 1)
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('N='+str(n))
    plt.savefig('%d.png'%n)
    
#    1-2
    plt.figure()
    plt.plot(x[:n],t[:n],'o')
    line2 = []
    for i in range(3):
        line2.append((line - mu[i])/0.1)
    line2 = np.concatenate((sigmoid(line2[0]).reshape(50,1),sigmoid(line2[1]).reshape(50,1),sigmoid(line2[2]).reshape(50,1)),axis=1)
    mx = []
    vx = []
    for point in line2:
        mx += [mn.T.dot(point)] #預測分佈的平均
        vx += [1. + (point.T.dot(sn)).dot(point)] #預測分佈的變異數
    mx = np.reshape(np.asarray(mx), len(mx))
    vx = np.reshape(np.asarray(vx), len(vx))
    plt.plot(line, mx, linestyle = '-', zorder = 1, color = 'red')
    plt.fill_between(line, mx-vx, mx+vx, color = 'pink') #範圍內塗色
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('N='+str(n))
    plt.savefig('pred_%d.png'%n)

#%%  
#    1-3
j1 = np.arange(0,2,1)
mu = j1*2/3
beta = 1
alpha = 10**-6
N = [5,10,30,80]
s = 0.1
a = []
for i in range(2):
    a.append((x - mu[i])/0.1)

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))
a1 = np.concatenate((sigmoid(a[0]),sigmoid(a[1])),axis=1)

for n in N:
    target = t[:n]
    PHI = a1[:n]
    s0_inv = (10**-6) * np.identity(2)
    sn_inv = s0_inv + PHI.T.dot(PHI)
    sn = np.linalg.inv(sn_inv)
    mn = sn.dot(PHI.T.dot(target))
    mn_ = np.reshape(mn, 2)
    w = np.linspace(-10, 10, 100)
    W = np.dstack(np.meshgrid(w, w))
    prior_vals = stats.multivariate_normal(mn_, sn).pdf(W)
    plt.figure()
    plt.contourf(w, w, prior_vals, 100)
    plt.xlabel('w0')
    plt.ylabel('w1')
    plt.title('N='+str(n))
    plt.savefig('a_%d.png'%n)