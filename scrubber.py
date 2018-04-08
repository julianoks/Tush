import os
import urllib
import numpy as np
import cv2
import torch
import torch.utils.data as data_utils
from math import ceil

WINE_DATA_URL="https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"


def get_tensors():
    wine_exists=False
    
    for fname in os.listdir("data"):
        if fname=="wine.txt":
            wine_exists=True
            
    wine_dir='data/wine.txt'
    
    if not wine_exists:
        urllib.urlretrieve(WINE_DATA_URL, wine_dir)
    
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
    
#     train_record_count=int(ceil(train_ratio*features.shape[0]))
# 
#     unique_class_count=3
#     
#     train=data_utils.TensorDataset(features[:train_record_count,:], classes[:train_record_count])
#     train_loader=data_utils.DataLoader(train, batch_size=1)
#     
#     valid=torch.utils.data.TensorDataset(features[train_record_count:,:], classes[train_record_count:])
#     valid_loader=torch.utils.data.DataLoader(train, batch_size=1)
    
    return features, classes

