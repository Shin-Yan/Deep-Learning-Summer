from dataloader import RetinopathyLoader
import torch
def main():
    train_dataset = RetinopathyLoader('./data', 'train')
    test_dataset = RetinopathyLoader('./data', 'test')
    train_device = torch.device("cuda" if cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

test_dataset[50]