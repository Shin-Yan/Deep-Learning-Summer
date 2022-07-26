from dataloader import RetinopathyLoader
import torch
def train():
    pass
def parsing():
    parser = ArgumentParser(description = 'parse the input arguments')
    parser.add_argument('-e',default = 10, type=int)
    parser.add_argument('-m', default = 'ResNet18', type = str)
    parser.add_argument('-lr', default = 1e-3, type = float)
    parser.add_argument('-b', default = 4, type = int)
    parser.add_argument('-o', default = 'sgd', type = str)
    parser.add_argument('-mo', default = 0.9, type = float)
    parser.add_argument('-c', default = 1, type = int)
    parser.add_argument('-p', default = 0, type = int)
    parser.add_argument('-l', default = 0, type = int)
    parser.add_argument('-w', default = 5e-4, type = float)
    return parser.parse_args()

def main():
    train_dataset = RetinopathyLoader('./data', 'train')
    test_dataset = RetinopathyLoader('./data', 'test')
    train_device = torch.device("cuda" if cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    train(model = arguments.m,epochs = arguments.e ,learning_rate = arguments.lr,batch_size = arguments.b,
            optimizer = optimizer,loss_function = torch.nn.CrossEntropyLoss(),momentum = arguments.mo,
            train_device = train_device,train_dataset = train_dataset,test_dataset = test_dataset, 
            pretrain = arguments.p, weight_decay = arguments.w)