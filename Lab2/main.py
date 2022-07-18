import dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn import Sequential, Conv2d, BatchNorm2d, AvgPool2d, Dropout, Flatten, Linear, LeakyReLU, ReLU, ELU

class EEGNet(torch.nn.Module):
    def __init__(self,activation,dropout):
        super().__init__()

        self.firstConv = Sequential(Conv2d(1, 16, kernel_size = (1,51), stride=(1,1), padding = (0,25), bias=False), 
                                    BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats =True))
        self.depthWiseConv = Sequential(Conv2d(16,32,kernel_size = (2,1), stride=(1,1),groups=16, bias=False),
                                        BatchNorm2d(32, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True),
                                        ELU(alpha =1.0),
                                        AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
                                        Dropout(p=0.25))
        self.separableConv = Sequential(Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
                                        BatchNorm2d(32, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True),
                                        ELU(alpha=1.0),
                                        AvgPool2d(kernel_size=(1,8),stride=(1,8),padding = 0),
                                        Dropout(p=0.25))
        self.classify = Sequential(Linear(in_features = 736, out_features =2 ,bias = True))
    def forwardPass(self, inputs):
        NetResult = self.separableConv(self.depthWiseConv(self.firstConv(inputs)))
        return self.classify(NetResult) 

class DeepConvNet(torch.nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()

def show_result(model, epochs, accuracy):
def train(model, epochs, learning_rate, batch_size, optimizer, loss_function, dropout, train_device,
            train_dataset, test_dataset):
    models = {}
    if(model == 'EEG'):
        models['ELU'] = EEGNet(ELU,dropout = dropout).to(train_device)
        models['ReLU'] = EEGNet(ReLU,dropout = dropout).to(train_device)
        models['LeakyReLU'] = EEGNet(LeakyReLU,dropout = dropout).to(train_device)
    elif(model == 'DeepConv'):
        models['ELU'] = DeepConvNet(ELU,dropout = dropout).to(train_device)
        models['ReLU'] = DeepConvNet(ReLU,dropout = dropout).to(train_device)
        models['LeakyReLU'] = DeepConvNet(LeakyReLU,dropout = dropout).to(train_device)
        
def main():
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    # print(train_data, train_label, test_data, test_label)
    print(type(train_data))
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if __name__ == '__main__':
    main()