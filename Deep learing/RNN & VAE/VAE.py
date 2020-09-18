from PIL import Image
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
import matplotlib.pyplot as plt

path = r'C:\Users\roger\OneDrive\桌面\HW2\data'
item = os.listdir(path)
df = []
for i in range(len(item)):
    im = Image.open(path +'/'+item[i] )   
    im = im.convert('RGB')
    im1 = np.asarray(im, dtype = np.uint8)
    im4 = Image.fromarray(im1,'RGB').resize((32,32))
    im1 = np.asarray(im4, dtype = np.uint8)
    df.append(im1)
df = np.array(df)
df = torch.from_numpy(df).float()
df = df/255
df = df.reshape([21551,3,32,32])
#%% 
image_size = 1024
#h_dim = 400
z_dim = 128
num_epochs = 501
batch_size = 144 
learning_rate = 1e-3

class VAE(nn.Module):
    def __init__(self, in_channels = 3,image_size=1024, hidden_dims=[32, 64, 128, 256, 512], z_dim=128):
        super(VAE, self).__init__()
        #self.fc1 = nn.Linear(image_size, hidden_dims[-1])
        self.fc2 = nn.Linear(hidden_dims[-1], z_dim) # 均值 向量mu
        self.fc3 = nn.Linear(hidden_dims[-1], z_dim) # 向量var
        self.fc4 = nn.Linear(z_dim, hidden_dims[-1]) #decoder_input
        Encoder_modules = []
        for hdim in hidden_dims:
            Encoder_modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=hdim,kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(hdim),
                    nn.LeakyReLU())
            )
            in_channels = hdim
        #print(*Encoder_modules)    
        self.encoder = nn.Sequential(*Encoder_modules)
        hidden_dims.reverse()
        Decoder_modules = []
        for i in  range(len(hidden_dims) - 1):
            Decoder_modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )
        self.decoder = nn.Sequential(*Decoder_modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels= 3,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())
    def encode(self, x):
        result = self.encoder(x)
        result = torch.flatten(result, start_dim=1)
        return self.fc2(result), self.fc3(result)
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        result = self.fc4(z)
        result = result.view(-1, 512, 1, 1)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result
    
    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconst = self.decode(z)
        return x_reconst, mu, log_var
model = VAE()   
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
df1 = df[:batch_size,:,:,:]
ELBO = []
for epoch in range(num_epochs):
    x = df1
    x_reconst, mu, log_var = model(x)
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    reconst_loss = F.mse_loss(x_reconst, x, reduction='sum')
    #loss = reconst_loss + kl_div
    #loss = reconst_loss + kl_div*100
    loss = reconst_loss + kl_div*0
    ELBO.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    with torch.no_grad():
        out, _, _ = model(x)
        x = x.reshape([batch_size,32, 32, 3])
        out = out.reshape([batch_size,32, 32, 3])
        #x_concat = torch.cat([x.reshape([-1, 32, 32, 3]), out.reshape([-1, 32, 32, 3])], dim=3)
        x = x.numpy()
        out = out.numpy()
    if epoch % 50 == 0:
        print(loss)
with torch.no_grad():
    z = torch.randn(batch_size, z_dim)
    out1 = model.decode(z).reshape([batch_size,32, 32, 3])
for i in range(batch_size):   
    im = Image.fromarray(np.uint8(out1[i,:,:,:]*255)) 
    plt.subplot(12,12,i+1)    
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
plt.show()    
for i in range(batch_size):   
    im = Image.fromarray(np.uint8(x[i,:,:,:]*255)) 
    plt.subplot(12,12,i+1)    
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
plt.show()
for i in range(batch_size):   
    im = Image.fromarray(np.uint8(out[i,:,:,:]*255)) 
    plt.subplot(12,12,i+1)    
    plt.imshow(im)
    plt.xticks([])
    plt.yticks([])
plt.show()
#%%
epoch = [i for i in range(num_epochs)]
plt.figure()
plt.plot(epoch, ELBO)
plt.xlabel('epoch')
plt.ylabel('loss')
plt.xticks(range(0,len(ELBO)+1,100))
plt.title('learning curve(2-2)')
plt.show()   
#%%
with torch.no_grad():
    mu1 = model.encode(df1[:2,:,:,:])[0][0]
    mu2 = model.encode(df1[:2,:,:,:])[0][1]
    v1 = model.encode(df1[:2,:,:,:])[1][0]
    v2 = model.encode(df1[:2,:,:,:])[1][1]
    recon1 = model.reparameterize(mu1,v1)
    recon2 = model.reparameterize(mu2,v2)
    dif = recon1-recon2
    interpolation = torch.zeros([1, 128], dtype=torch.float64)
    outt = model.decode(dif).reshape([32, 32, 3]) 
    outt = outt.numpy()
im = Image.fromarray(np.uint8(outt*255)) 
plt.imshow(im)
plt.xticks([])
plt.yticks([])

