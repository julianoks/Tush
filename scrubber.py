import os
import urllib
import numpy as np
import cv2
import torch
import torch.utils.data as data_utils
from torchvision import datasets, transforms
from math import ceil


class data_wrangler(object):
    '''
    Datasets available at the moment:
    * Wine-wine
    * MNIST-mnist
    '''
    def __init__(self):
        self.wine_data_url="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"





    def get_wine_loaders(self, train_ratio=0.7):
        wine_exists=False
    
        for fname in os.listdir("data"):
            if fname=="wine.txt":
                wine_exists=True
            
        wine_dir='data/wine.txt'
    
        if not wine_exists:
            urllib.urlretrieve(self.wine_data_url, wine_dir)
            
            
        classes=[]
        features=[]
        with open(wine_dir) as fp:
            for i, line in enumerate(fp):
                if line[-1]=='\n':
                    line=line[:-1]
                tensor_full=line.split(",")
                
                classes.append(int(tensor_full[0])-1)
                features.append(map(lambda x: float(x), tensor_full[1:]))
        torch.manual_seed(1)
        features=torch.FloatTensor(features)
        features=features[torch.randperm(features.shape[0])]
        torch.manual_seed(1)
        classes=torch.IntTensor(classes)
        classes=classes[torch.randperm(classes.shape[0])]
        
        assert features.shape[0]==classes.shape[0], "Number of records in classes and features mismatch"
        
        train_record_count=int(ceil(train_ratio*features.shape[0]))

        unique_class_count=3
        
        train_features=features[:train_record_count, :]
        train_classes=classes[:train_record_count]
        
        valid_features=features[train_record_count:,:]
        valid_classes=classes[train_record_count:]
        
        batches={
        'train': self.make_tensor_batches(train_features, train_classes, unique_class_count, max_batches=train_classes.shape[0]),
        'validation': self.make_tensor_batches(valid_features, valid_classes, unique_class_count, max_batches=valid_classes.shape[0])
        }
        
        
        return batches


    def get_mnist_loaders(self):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=True)

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False,
                           transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=True)
        
        batches = {
            'train': self.make_dataloader_batches(train_loader, 10, max_batches=100000),
            'validation': self.make_dataloader_batches(test_loader, 10, max_batches=10000),
        }
        
        return batches
        
        
        
    def make_dataloader_batches(self, loader, shape, max_batches=None, use_cuda=False):
        batch = []
        for data, target in loader:
            if use_cuda: data, target = data.cuda(), target.cuda()
            data, target = torch.autograd.Variable(data), target[0]
            batch.append([[[[['tensor', data]], [shape]]], [target]])
            if len(batch) >= max_batches:
                break
        return batch


    def make_tensor_batches(self, features, classes, shape, max_batches=None, use_cuda=False):
        batch = []
        for i in range(features.shape[0]):
            if use_cuda: data, target=features[i,:].cuda(), classes[i].cuda()
            data, target=torch.autograd.Variable(features[i,:]), classes[i]
            batch.append([[[[['tensor', data]], [shape]]], [target]])
            if len(batch) >= max_batches:
                break
        return batch





    

