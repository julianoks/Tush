import cv2
import tush
import programmer
import torch
import torch.utils.data as data_utils
from scrubber import data_wrangler
import numpy as np
from math import ceil



batches=data_wrangler().get_mnist_loaders()


def loss_fn(pred, target_idx):
    return - torch.log(torch.nn.functional.softmax(pred)[target_idx])

program = [
      ['exec', 'add'],
      ['blueprint', [['exec', 'folded_normal'], ['exec', 'shape_1d'], ['integer', 10]]],
      ['exec', 'relu'],
      ['exec', 'matmul_backward'],
      ['blueprint', [['exec', 'folded_normal'], ['exec', 'shape_3d'], ['integer', 28], ['integer', 28], ['integer', 10]]],
      ]

ind = tush.Tush(program)
ind.stage_two(batches['train'], loss_fn)
results = ind.stage_three(validation_batch=batches['validation'], loss_fn=loss_fn)
print("Results:", results)