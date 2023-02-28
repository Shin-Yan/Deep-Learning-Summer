import torch
import torch.nn as nn

def weights_init(model):

    classname = model.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0)
        
class Generator(nn.Module):
    def __init__(self, noise_size, cond_size):
        super(Generator,self).__init__()
        self.noise_size = noise_size
        self.cond_size = cond_size

        self.conditionExpand = nn.Sequential(
            nn.Linear(24, cond_size),
            nn.ReLU()
        )


        self.convG1 = nn.Sequential(
            nn.ConvTranspose2d(noise_size + cond_size, 512, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True)
            )
        self.convG2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
            )
        self.convG3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
            )
        self.convG4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True)
            )
        self.convG5 = nn.ConvTranspose2d(64, 3, kernel_size=4 , stride=2, padding=1, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, noise, cond):
        
        noise = noise.view(-1,self.noise_size,1,1)
        cond = self.conditionExpand(cond).view(-1, self.cond_size, 1, 1)
        out=torch.cat((noise, cond), dim=1)  # (N, noise+cond, 1, 1)
        out=self.convG1(out)  # (N, 512, 4, 4)
        out=self.convG2(out)  # (N, 256, 8, 8)
        out=self.convG3(out)  # (N, 128, 16, 16)
        out=self.convG4(out)  # (N, 64, 32, 32)
        out=self.convG5(out)  # (N, 3, 64, 64)
        out=self.tanh(out)    # set value between [-1,+1]
        return out


class Discriminator(nn.Module):
    def __init__(self, img_shape, cond_size):
        super(Discriminator, self).__init__()
        self.H, self.W, self.C = img_shape

        self.conditionExpand = nn.Sequential(
            nn.Linear(24, self.H * self.W * 1),
            nn.LeakyReLU()
        )

        self.convD1 = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.convD2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.convD3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.convD4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
            )
        self.convD5 = nn.Conv2d(512, 1, kernel_size=4, stride=1,padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X, cond):

        cond = self.conditionExpand(cond).view(-1, 1, self.H, self.W)
        out=torch.cat((X, cond), dim=1)  # (N, 4, 64, 64)
        out=self.convD1(out)  # (N, 64, 32, 32)
        out=self.convD2(out)  # (N, 128, 16, 16)
        out=self.convD3(out)  # (N, 256, 8, 8)
        out=self.convD4(out)  # (N, 512, 4, 4)
        out=self.convD5(out)  # (N, 1, 1, 1)
        out=self.sigmoid(out)  # set value between [0,1]
        out=out.view(-1)
        return out