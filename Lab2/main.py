import dataloader
import torch
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from torch.nn import Sequential, Conv2d, BatchNorm2d, AvgPool2d, MaxPool2d, Dropout, Flatten, Linear, LeakyReLU, ReLU, ELU
from tqdm import tqdm
from argparse import ArgumentParser

class EEGNet(torch.nn.Module):
    def __init__(self,activation):
        super().__init__()

        self.firstConv = Sequential(Conv2d(1, 16, kernel_size = (1,51), stride=(1,1), padding = (0,25), bias=False), 
                                    BatchNorm2d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats =True))
        self.depthWiseConv = Sequential(Conv2d(16,32,kernel_size = (2,1), stride=(1,1),groups=16, bias=False),
                                        BatchNorm2d(32, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True),
                                        activation(),
                                        AvgPool2d(kernel_size=(1,4),stride=(1,4),padding=0),
                                        Dropout(p=0.25))
        self.separableConv = Sequential(Conv2d(32,32,kernel_size=(1,15),stride=(1,1),padding=(0,7),bias=False),
                                        BatchNorm2d(32, eps=1e-05, momentum=0.1, affine = True, track_running_stats=True),
                                        activation(),
                                        AvgPool2d(kernel_size=(1,8),stride=(1,8),padding = 0),
                                        Dropout(p=0.25))
        self.classify = Sequential(Flatten(),Linear(in_features = 736, out_features =2 ,bias = True))
    def forwardPass(self, inputs):
        NetResult = self.separableConv(self.depthWiseConv(self.firstConv(inputs)))
        return self.classify(NetResult) 

class DeepConvNet(torch.nn.Module):
    def __init__(self, activation, dropout):
        super().__init__()
        self.Conv1 = Sequential(Conv2d(1,25,kernel_size= (1,5), bias = False),
                                Conv2d(25,25,kernel_size = (2,1), bias = False),
                                BatchNorm2d(25, eps=1e-05, momentum=0.1),
                                activation(),
                                MaxPool2d(kernel_size = (1,2)),
                                Dropout(p=dropout))
        self.Conv2 = Sequential(Conv2d(25,50,kernel_size= (1,5), bias = False),
                                BatchNorm2d(50, eps=1e-05, momentum=0.1),
                                activation(),
                                MaxPool2d(kernel_size = (1,2)),
                                Dropout(p=dropout))
        self.Conv3 = Sequential(Conv2d(50,100,kernel_size= (1,5), bias = False),
                                BatchNorm2d(100, eps=1e-05, momentum=0.1),
                                activation(),
                                MaxPool2d(kernel_size = (1,2)),
                                Dropout(p=dropout))
        self.Conv4 = Sequential(Conv2d(100,200,kernel_size= (1,5), bias = False),
                                BatchNorm2d(200, eps=1e-05, momentum=0.1),
                                activation(),
                                MaxPool2d(kernel_size = (1,2)),
                                Dropout(p=dropout))
        flatten_size = 8600
        output_size = 2
        self.classify = Sequential(Flatten(),Linear(in_features = flatten_size, out_features = output_size,bias = True))
    def forwardPass(self, inputs):
        for i in range(1,5):
            inputs = getattr(self, f'Conv{i}')(inputs)
        return self.classify(inputs)

def show_result(model, epochs, accuracy_test, accuracy_train):
    plt.figure(0)
    if model == 'EEG':
        plt.title('Activation function comparison(EEGNet)')
    else:
        plt.title('Activation function comparison(DeepConvNet)')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    for model_name in accuracy_train.keys():
        plt.plot(range(epochs), accuracy_train[model_name], label=f'{model_name}')
        print(f'{model_name}: {max(accuracy_train[model_name]):.2f} %')
    for model_name in accuracy_test.keys():
        plt.plot(range(epochs), accuracy_test[model_name], label=f'{model_name}')
        print(f'{model_name}: {max(accuracy_test[model_name]):.2f} %')
    plt.legend(loc='lower right')
    plt.savefig('comparison_result.png')
    
def train(model, epochs, learning_rate, batch_size, optimizer, loss_function, dropout, train_device,
          train_dataset, test_dataset):
    models = {}
    if(model == 'EEG'):
        models['elu'] = EEGNet(ELU).to(train_device)
        models['relu'] = EEGNet(ReLU).to(train_device)
        models['leakyrelu'] = EEGNet(LeakyReLU).to(train_device)
    elif(model == 'DeepConv'):
        models['elu'] = DeepConvNet(ELU,dropout = dropout).to(train_device)
        models['relu'] = DeepConvNet(ReLU,dropout = dropout).to(train_device)
        models['leakyrelu'] = DeepConvNet(LeakyReLU,dropout = dropout).to(train_device)
    accuracy_test = {key+'_test':[0]*epochs for key in models.keys()}
    accuracy_train = {key + '_train':[0]*epochs for key in models.keys()}
    
    # Start training
    train_loader = DataLoader(train_dataset, batch_size = batch_size)
    test_loader = DataLoader(test_dataset, batch_size = len(test_dataset))
    for key in models.keys():
        model_optimizer = optimizer(models[key].parameters(), lr = learning_rate)
        print(f"Training {model} with activation {key}")
        for epoch in tqdm(range(epochs)):
            # Train
            models[key].train()
            tmp_hit = 0
            for data, label in train_loader:
                # put data to GPU if GPU is available
                model_inputs = data.to(train_device)
                model_label = label.to(train_device).long()
                
                model_prediction = models[key].forwardPass(inputs = model_inputs)
                model_optimizer.zero_grad() # set gradient of ortimizer to zero
                loss = loss_function(model_prediction, model_label)
                loss.backward()
                model_optimizer.step()
                tmp_hit += (torch.max(model_prediction, 1)[1] == model_label).sum().item()
            accuracy_train[key + '_train'][epoch] = tmp_hit/len(train_dataset)*100

            # Test
            # switch to testing mode
            models[key].eval()
            tmp_hit = 0
            # disable gradient computing
            with torch.no_grad():
                for data, label in test_loader:
                # put data to GPU if GPU is available
                    model_inputs = data.to(train_device)
                    model_label = label.to(train_device).long()
                    
                    model_prediction = models[key].forwardPass(inputs = model_inputs)
                    tmp_hit += (torch.max(model_prediction, 1)[1] == model_label).sum().item()
                accuracy_test[key + '_test'][epoch] = tmp_hit/len(test_dataset)*100
        torch.cuda.empty_cache()    
    show_result(model,epochs,accuracy_test,accuracy_train)
def parsing():
    parser = ArgumentParser(description = 'parse the input arguments')
    parser.add_argument('-e',default = 300, type=int)
    parser.add_argument('-m', default = 'EEG', type = str)
    parser.add_argument('-lr', default = 1e-2, type = float)
    parser.add_argument('-b', default = 64, type = int)
    parser.add_argument('-o', default = 'adam', type = str)
    parser.add_argument('-d', default = 0.5, type = float)
    return parser.parse_args()

def get_optimizer(opt_str):
    if opt_str == 'adam':
        return torch.optim.Adam
    elif opt_str == 'adadelta':
        return torch.optim.Adadelta
    elif opt_str == 'adagrad':
        return torch.optim.Adagrad
    elif opt_str == 'adamw':
        return torch.optim.AdamW
    elif opt_str == 'adamax':
        return torch.optim.Adamax
    elif opt_str == 'sparseadam':
        return torch.optim.SparseAdam
    elif opt_str == 'asgd':
        return torch.optim.ASGD
    elif opt_str == 'lbfgs':
        return torch.optim.LBFGS
    elif opt_str == 'nadam':
        return torch.optim.NAdam
    elif opt_str == 'radam':
        return torch.optim.RAdam
    elif opt_str == 'rmsprop':
        return torch.optim.RMSprop
    elif opt_str == 'rprop':
        return torch.optim.Rprop
    elif opt_str == 'sgd':
        return torch.optim.SGD

    raise ArgumentTypeError(f'Optimizer {opt_str} is not supported.')

def main():
    arguments = parsing()
    print(arguments)
    train_data, train_label, test_data, test_label = dataloader.read_bci_data()
    
    train_dataset = TensorDataset(torch.Tensor(train_data), torch.Tensor(train_label))
    test_dataset = TensorDataset(torch.Tensor(test_data), torch.Tensor(test_label))

    train_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #get the optimizer
    optimizer = get_optimizer(arguments.o)
    torch.cuda.set_device(1)
    train(model = arguments.m,epochs = arguments.e ,learning_rate = arguments.lr,batch_size = arguments.b,
            optimizer = optimizer,loss_function = torch.nn.CrossEntropyLoss(),dropout=0.25,
            train_device = train_device,train_dataset = train_dataset,test_dataset = test_dataset)
if __name__ == '__main__':
    main()