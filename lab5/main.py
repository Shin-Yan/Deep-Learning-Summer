import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as dset
import torchvision.models as models
import torchvision
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from torch.autograd import Variable
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

from data_loader import iclevrDataSet
from model import Generator, Discriminator, weights_init
from evaluator import evaluation_model

# hyperparameters 
batch_size = 32
lr = 2e-4
n_epoch = 250
image_shape = (64, 64, 3)
noise_size = 100
cond_size = 200
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# model

G = Generator(noise_size, cond_size).to(device)
D = Discriminator(image_shape, cond_size).to(device)

G.apply(weights_init)
D.apply(weights_init)

# loss criterion
criterion = nn.BCELoss()

# optimizer
opt_D = torch.optim.Adam(D.parameters(), lr=lr,betas=(0.5, 0.999))
opt_G = torch.optim.Adam(G.parameters(), lr=lr,betas=(0.5, 0.999))
evaluation_model = evaluation_model()

# dataset
dataset = iclevrDataSet()
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# test conditions
with open('objects.json', 'r') as file:
    obj_dict = json.load(file)

    
with open('new_test.json','r') as file:
    test_dict = json.load(file)
    n_test = len(test_dict)
    
test_cond = torch.zeros(n_test, 24)

for i in range(n_test):
    for condition in test_dict[i]:
        test_cond[i, obj_dict[condition]] = 1.
        

test_noise = torch.randn(n_test, noise_size) # sample noise from normal distribution
test_noise = test_noise.to(device)
test_cond = test_cond.to(device)
#print('shape:',test_cond.shape)


def train():
    best_score = 0
    for epoch in range(1, n_epoch+1):
        total_loss_G = 0
        total_loss_D = 0
        for i, (images, conditions) in enumerate(dataloader):
            G.train()
            D.train()
            batch_size = len(images)
            images = images.to(device)
            conditions = conditions.to(device)
            
            real = torch.ones(batch_size).to(device)
            fake = torch.zeros(batch_size).to(device)
            
            # Train discriminator
            opt_D.zero_grad()
            
            # Real 
            predicts = D(images.detach(), conditions)
            loss_real = criterion(predicts, real)
            
            # Fake
            z = torch.randn(batch_size, noise_size).to(device)
            gen_imgs = G(z, conditions)
            predicts = D(gen_imgs.detach(), conditions)
            loss_fake = criterion(predicts, fake)
            
            # Update Model
            loss_D = loss_real + loss_fake
            loss_D.backward()
            opt_D.step()
            
            
            # Train generator
            for _ in range(4):
                opt_G.zero_grad()
                
                z = torch.randn(batch_size, noise_size).to(device)
                gen_imgs = G(z, conditions)
                predicts = D(gen_imgs, conditions)
                loss_G = criterion(predicts, real)
                
                # Update Model
                loss_G.backward()
                opt_G.step()
            
            print(f'\rEpoch[{epoch}/{n_epoch}] {i+1}/{len(dataloader)}  Loss_G: {loss_G.item():.4f}  Loss_D: {loss_D.item():.4f}', end='')
            total_loss_G += loss_G.item()
            total_loss_D += loss_D.item()
        
        # Evaluate
        G.eval()
        D.eval()
        with torch.no_grad():
            gen_imgs = G(test_noise, test_cond)
        score = evaluation_model.eval(gen_imgs, test_cond)
        print(f'\nScore: {score:.2f}')

        if score > best_score:
            print('Parameters saved!\n')
            torch.save(G.state_dict(), 'G_weight.pth')
            torch.save(D.state_dict(), 'D_weight.pth')
            torchvision.utils.save_image(gen_imgs, 'result.png', nrow=8, normalize=True)
            best_score = score
            
            

def test():    

    G = Generator(noise_size, cond_size).to(device)
    G.load_state_dict(torch.load('G_weight.pth'))

    with torch.no_grad():
        gen_imgs = G(test_noise, test_cond) 
        score = evaluation_model.eval(gen_imgs, test_cond)
        print(f'\nScore: {score:.2f}')

    # show generated image
    grid_img = torchvision.utils.make_grid(gen_imgs.cpu(),nrow=8, normalize=True)
    plt.figure(figsize=(8, 4))
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.show()


if __name__ == "__main__":
    
    train()
    # test()
        