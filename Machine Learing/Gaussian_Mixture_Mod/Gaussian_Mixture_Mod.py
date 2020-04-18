#%%
# =============================================================================
# 3.Gaussian Mixture Mod
# =============================================================================

import numpy as np
from numpy.linalg import inv, det, pinv
import matplotlib.pyplot as plt
import cv2

img = cv2.imread('hw3_3.jpeg')

shrunk = 0.5 
img = cv2.resize(img, (int(img.shape[1]*shrunk),int(img.shape[0]*shrunk)), interpolation=cv2.INTER_CUBIC)
img = img/255

def k_means(pic,k,save):
    np.random.seed(123)
    channel = 3
    means = np.random.rand(k, channel)
    while True:
        tag_map = np.zeros((pic.shape[0], pic.shape[1]))
        new_means = np.zeros((k, channel))
        cnt = np.zeros((k))
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                min_dist = 1000000.
                tag = -1
                for j in range(k):
                    if (pic[r][c] - means[j]).dot(pic[r][c] - means[j]) < min_dist:
                        min_dist = (pic[r][c] - means[j]).dot(pic[r][c] - means[j])
                        tag = j
                tag_map[r][c] = tag
                new_means[tag] += pic[r][c]
                cnt[tag] += 1
        for j in range(k):
            if cnt[j] == 0:
                new_means[j] = 0.
            else:
                new_means[j] /= float(cnt[j])
        if (np.absolute(new_means - means) < np.ones((k, channel)) * 0.003).all():
            break
        new_var = np.zeros((k, channel, channel))
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                for j in range(k):
                    new_var[j]+=float(tag_map[r][c] == j) * (img[r][c].reshape(channel,1) - new_means[j].reshape(channel,1)).dot((img[r][c].reshape(channel,1) - new_means[j].reshape(channel,1)).T)
        for j in range(k):
            if cnt[j] == 0:
                new_var[j] = np.zeros((channel, channel))
            else:
                new_var[j] /= float(cnt[j])
        means = np.copy(new_means)
        kvar = np.copy(new_var)
        kmeans = np.copy(means)
        kpi = cnt / float(pic.shape[0]*pic.shape[1])
    if save == 1:
        tmp = np.copy(pic)
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                tmp[r][c] = means[int(tag_map[r][c])] * 255.
        cv2.imwrite('kmean_%d.jpg'%k, tmp)
    return kmeans, kpi, kvar
k1 = [3,5,7,10]
km = []
for i in k1: 
   km.append(k_means(img,i,1)[0])
#%%
def gmm(pic, k, save):
    def normal_d(x, mean, cov):
        if det(cov) != 0:
            return ((2*np.pi)**(-1*k/2.)) * (det(cov)**-0.5) * np.exp(-0.5 * ((x - mean).T.dot(inv(cov)).dot(x - mean)))
        else:
            cov = cov + 0.0001*np.identity(3)
            return ((2*np.pi)**(-1*k/2.)) * (det(cov)**-0.5) * np.exp(-0.5 * ((x - mean).T.dot(inv(cov)).dot(x - mean)))
    def log_likelihood(pic, pi, mean, var):
        log_like = 0.
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                tmp = 0.
                xn = pic[r][c].reshape(channel,1)
                for j in range(k):
                    tmp += pi[j] * normal_d(xn, mean[j].reshape(channel,1), var[j])
                log_like += np.log(tmp)
        return log_like
    channel = 3
    #init
    means, pi, var = k_means(pic,k,save)

    #training
    rnk = np.zeros((pic.shape[0], pic.shape[1],k))
    epoch = 0
    ex = []
    lx = []
    while True:
        lk = log_likelihood(pic, pi, means, var)
        print (epoch, lk)
        ex += [epoch]
        lx += [lk.reshape(1)]
        nk = np.zeros(k)
        new_means = np.zeros((k, channel))
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                xn = pic[r][c].reshape(channel,1)
                mix_gau = np.zeros((k, 1))
                for j in range(k):
                    mix_gau[j] = pi[j] * normal_d(xn, means[j].reshape(channel,1), var[j])
                sum_ = np.sum(mix_gau)
                for j in range(k):
                    rnk[r][c][j] = mix_gau[j] / sum_
                    nk[j] += rnk[r][c][j]
                    new_means[j] += rnk[r][c][j] * xn.reshape(channel)
        for j in range(k):
            new_means[j] = np.copy(new_means[j] / nk[j])
        means = new_means #update means
        new_var = np.zeros((k, channel, channel))
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                xn = pic[r][c].reshape(channel,1)
                for j in range(k):
                    new_var[j] += rnk[r][c][j] * (xn - means[j].reshape(channel,1)).dot((xn - means[j].reshape(channel,1)).T)
        new_pi = np.zeros(k)
        for j in range(k):
            new_var[j] /= nk[j]
            new_pi[j] = nk[j] / float(pic.shape[0]*pic.shape[1])
        pi = np.copy(new_pi)
        var = np.copy(new_var)
        epoch += 1
        if epoch > 100:
            break
    gmean = means
    gvar = var
    if save == 1:
        tmp = np.copy(pic)
        for r in range(pic.shape[0]):
            for c in range(pic.shape[1]):
                xn = tmp[r][c].reshape(channel,1)
                p = np.zeros(k)
                for j in range(k):
                    p[j] = normal_d(xn, gmean[j].reshape(channel,1), gvar[j])
                tmp[r][c] = gmean[np.argmax(p)] * 255.
        cv2.imwrite('gmm_%d.jpg'%k, tmp)
    plt.figure()
    plt.plot(ex, lx, linestyle = '--')
    plt.title('log likelihood curve ( k = '+str(k)+' )')
    plt.savefig('log_likelihood_%d.png' %k)
for i in k1:
    gmm(img,i,1)