#%% 
# =============================================================================
# 1.Gaussian Process for Regression
# =============================================================================
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt
import scipy.io as io
import os
os.chdir(r'C:\Users\517-B\Desktop\ML_HW3')
data = io.loadmat('gp.mat')['x']
label = io.loadmat('gp.mat')['t']
x_train = data[0:60]
x_test = data[60:100]
t_train = label[0:60]
t_test = label[60:100]
class kernel:
    def __init__(self, t0, t1, t2, t3):
        self.t0 = float(t0)
        self.t1 = float(t1)
        self.t2 = float(t2)
        self.t3 = float(t3)
    def k(self,xn, xm):
        return  self.t0 * np.exp(-0.5*self.t1*(xn-xm).dot(xn-xm)) + self.t2 + self.t3*xn.T.dot(xm)

class gaussian_process:
    def __init__(self,t):
        self.theta = np.asarray(t).reshape(4,1)
        self.k = kernel(t[0],t[1],t[2],t[3])
    def fit(self, x, t):
        self.t = t
        self.x = x
        self.C = np.zeros((len(self.x),len(self.x)))
        for n in range(len(self.x)):
            for m in range(len(self.x)):
                self.C[n][m] = self.k.k(x[n],x[m]) + float(n == m) 
    def predict(self, data1): 
        ka = np.zeros((len(self.x),1))
        for n in range(len(self.x)):
            ka[n] = self.k.k(self.x[n],data1)
        mean = ka.T.dot(inv(self.C)).dot(self.t)
        var = (self.k.k(data1,data1) + 1.0) - ka.T.dot(inv(self.C)).dot(ka)
        return mean, var
    def rms(self, datas, target):
        e = 0.
        for i,data in enumerate(datas):
            mean, var = self.predict(data)
            e += ((mean - target[i])**2)
        e /= len(datas)
        return np.sqrt(e)
    def ard(self, lr): 
        def c_diff(self, term):
            dc = np.zeros((len(self.x),len(self.x)))
            for n in range(len(self.x)):
                for m in range(len(self.x)):
                    if term == 0:
                        dc[n][m] = np.exp(-0.5*self.theta[1]*((self.x[n] - self.x[m])**2))
                    elif term == 1:
                        dc[n][m] = self.theta[0] * np.exp(-0.5*self.theta[1]*((self.x[n] - self.x[m])**2)) * (-0.5**((self.x[n] - self.x[m])**2))
                    elif term == 2: 
                        dc[n][m] = 1.
                    else: 
                        dc[n][m] = self.x[n].T.dot(self.x[m])
            return dc
        epoch = 0
        ex = []
        while True:
            ex += [epoch]
            update = np.zeros((4,1))
            flag = 0
            for i in range(4):
                update[i] = -0.5*np.trace(inv(self.C).dot(c_diff(self,i))) + 0.5*self.t.T.dot(inv(self.C)).dot(c_diff(self,i)).dot(inv(self.C)).dot(self.t)
                if np.absolute(update[i]) < 6.:
                    flag += 1
            self.theta += lr*update
            self.k = kernel(self.theta[0][0],self.theta[1][0],self.theta[2][0],self.theta[3][0])
            self.C = np.zeros((len(self.x),len(self.x)))
            for n in range(len(self.x)):
                for m in range(len(self.x)):
                    self.C[n][m] = self.k.k(self.x[n],self.x[m]) + float(n == m)
            epoch += 1
            if flag == 4:
                break

theta = [[0,0,0,1],[1,4,0,0],[1,4,0,5],[1,32,5,5]]
RMS = []
for i in theta:
    gp = gaussian_process(i)
    gp.fit(x_train, t_train)

    line = np.linspace(0.,2.,100).reshape(100,1)
    mx = []
    vx = []
    for sample in line:
        mean, var =  gp.predict(sample)
        mx += [mean]
        vx += [var]
    mx = np.asarray(mx).reshape(100,1)
    vx = np.asarray(vx).reshape(100,1)
    plt.plot(x_train, t_train,'bo')
    plt.plot(line, mx, linestyle = '-', color = 'red')
    plt.fill_between(line.reshape(100), (mx-vx).reshape(100), (mx+vx).reshape(100), color = 'pink')
    plt.title('θ = [ '+str(i[0])+' , '+str(i[1])+' , '+str(i[2])+' , '+str(i[3])+' ]')
    plt.savefig(str(i)+'.png')
    plt.show()
    
    RMS.append([gp.rms(x_train, t_train)[0][0],gp.rms(x_test, t_test)[0][0]])
print (RMS)
#%%1-4
theta = [3.,6.,4.,5.]
gp = gaussian_process(theta)
gp.fit(x_train, t_train)
gp.ard(0.001)
theta1 = list(np.round(gp.theta.reshape(1,-1)[0],2))
line = np.linspace(0.,2.,100).reshape(100,1)
mx = []
vx = []
for sample in line:
    mean, var =  gp.predict(sample)
    mx += [mean]
    vx += [var]
mx = np.asarray(mx).reshape(100,1)
vx = np.asarray(vx).reshape(100,1)
plt.plot(x_train, t_train,'bo')
plt.plot(line, mx, linestyle = '-', color = 'red')
plt.fill_between(line.reshape(100), (mx-vx).reshape(100), (mx+vx).reshape(100), color = 'pink')
plt.title('θ = [ '+str(theta1[0])+' , '+str(theta1[1])+' , '+str(theta1[2])+' , '+str(theta1[3])+' ]')
print (gp.rms(x_train, t_train)[0][0], gp.rms(x_test, t_test)[0][0],theta1)
RMS.append([gp.rms(x_train, t_train)[0][0], gp.rms(x_test, t_test)[0][0]])
plt.savefig('ard.png')
from pandas.core.frame import DataFrame
RMS = DataFrame(RMS)
RMS.rename(columns={0:'train',1:'test'},inplace=True)
RMS.rename(index={0:'[0,0,0,1]',1:'[1,4,0,0]',2:'[1,4,0,5]',3:'[1,32,5,5]',4:str(theta1)},inplace=True)
RMS.to_excel('rms.xlsx')