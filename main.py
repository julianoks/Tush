import tush, programmer
import torch

from torchvision import datasets, transforms
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('data', train=True, download=True,
		transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=1, shuffle=True)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST('data', train=False,
		transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])),batch_size=1, shuffle=True)

def make_batches(loader, shape, max_batches=None):
	batch = []
	for i, (data, target) in enumerate(loader):
		#if use_cuda: data, target = data.cuda(), target.cuda()
		data, target = torch.autograd.Variable(data), target[0]
		batch.append([[[[['tensor', data]], [10]]], [target]])
		if i==max_batches: break
	return batch

batches = {
	'train': make_batches(train_loader, [10], max_batches=1000),
	'validation': make_batches(test_loader, [10], max_batches=10000),
}

def loss_fn(pred, target_idx):
	return - torch.log(torch.nn.functional.softmax(pred)[target_idx])

program = [['exec', 'matmul'], ['exec', 'add'], ['exec', 'relu'],
	['blueprint', [['integer', 28], ['integer', 28], ['integer', 10], ['exec', 'shape_3d'], ['exec', 'folded_normal']]],
	['blueprint', [['integer', 10], ['exec', 'shape_1d'], ['exec', 'folded_normal']]],
	]

ind = tush.Tush(program)
val_loss = ind.optimize(batches['train'], loss_fn, validation_batch=batches['validation'])
print("Validation loss", val_loss)
