import cv2
import tush
import programmer
import torch

from torchvision import datasets, transforms
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False,
                   transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])), batch_size=1, shuffle=True)


def make_batches(loader, shape, max_batches=None, use_cuda=False):
    batch = []
    for data, target in loader:
        if use_cuda: data, target = data.cuda(), target.cuda()
        data, target = torch.autograd.Variable(data), target[0]
        batch.append([[[[['tensor', data]], [10]]], [target]])
        if len(batch) >= max_batches:
            break
    return batch

batches = {
    'train': make_batches(train_loader, [10], max_batches=100000),
    'validation': make_batches(test_loader, [10], max_batches=10000),
}


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