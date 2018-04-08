import cv2
import tush
import programmer
import torch
import torch.utils.data as data_utils
import scrubber
import numpy as np
from math import ceil


TRAIN_RATIO=0.7


features, classes=scrubber.get_tensors()

train_record_count=int(ceil(TRAIN_RATIO*features.shape[0]))

unique_class_count=3

train_features=features[:train_record_count, :]
train_classes=classes[:train_record_count]

valid_features=features[train_record_count:,:]
valid_classes=classes[train_record_count:]


def make_batches(features, classes, shape, max_batches=None, use_cuda=False):
    batch = []
    for i in range(features.shape[0]):
        if use_cuda: data, target=features[i,:].cuda(), classes[i].cuda()
        data, target=torch.autograd.Variable(features[i,:]), classes[i]
        batch.append([[[[['tensor', data]], [unique_class_count]]], [target]])
        if len(batch) >= max_batches:
            break
    return batch
    
batches={
    'train': make_batches(train_features, train_classes, unique_class_count, max_batches=train_classes.shape[0]),
    'validation': make_batches(valid_features, valid_classes, unique_class_count, max_batches=valid_classes.shape[0])
    }

def loss_fn(pred, target_idx):
    return - torch.log(torch.nn.functional.softmax(pred)[target_idx])

program = [
      ['exec', 'add'],
      ['blueprint', [['exec', 'folded_normal'], ['exec', 'shape_1d'], ['integer', unique_class_count]]],
      ['exec', 'relu'],
      ['exec', 'matmul_backward'],
      ['blueprint', [['exec', 'folded_normal'], ['exec', 'shape_2d'], ['integer', 6], ['integer', 3]]],
      ['exec', 'relu'],
      ['exec', 'matmul_backward'],
      ['blueprint', [['exec', 'folded_normal'], ['exec', 'shape_2d'], ['integer', int(features.shape[1])], ['integer', 6]]],
      ]

ind = tush.Tush(program)
ind.stage_two(batches['train'], loss_fn)
results = ind.stage_three(validation_batch=batches['validation'], loss_fn=loss_fn)
print("Results:", results)