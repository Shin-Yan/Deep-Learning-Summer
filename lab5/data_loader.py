import pandas as pd
import numpy as np
import json
import torch
from PIL import Image
from torch.utils import data
import torchvision.transforms as transforms

class iclevrDataSet(data.Dataset):
    def __init__(self):
        with open('train.json','r') as file:
            file_dict = json.load(file)
        
        self.length = len(file_dict)
        # "gray cube": 0
        self.img_name = list(file_dict.keys())
        self.labels = list(file_dict.values())
        self.transformations = transforms.Compose([transforms.Resize((64,64)), 
                                                   transforms.ToTensor(),
                                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        img_path = './iclevr/' + self.img_name[index]
        img = Image.open(img_path)
        img = img.convert('RGB')
        img = self.transformations(img)
        labels = self.labels[index]
        label = []
        # Turn into one-hot
        with open('objects.json', 'r') as file:
            obj_dict = json.load(file)
        
        for i in labels:
            label.append(obj_dict[i])
        labels = torch.zeros(24)
        
        for i in label:
            labels[i] = 1.0
        
        return img, labels
        