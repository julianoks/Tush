import torch
import utils

def get_shape(stacks):
	if stacks['tensor']:
		return stacks['tensor'][0]['val'].shape
	raise ValueError

Instructions = {
	
	'matmul': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.mm(a,b),
		'stochastic': False
	},

	'add': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.add(a,b),
		'stochastic': False
	},

	'relu': {
		'in_types': ['tensor'],
		'out_type': 'tensor',
		'fn':  lambda a: torch.nn.functional.relu(a),
		'stochastic': False
	},

	'folded_normal': {
		'in_types': ['shape'],
		'out_type': 'tensor',
		'fn': lambda shape: torch.autograd.Variable(torch.abs(torch.normal(torch.zeros(shape))) * 0.01, requires_grad=False),
		'stochastic': True
	},

	'uniform': {
		'in_types': ['shape'],
		'out_type': 'tensor',
		'fn': lambda shape: torch.autograd.Variable(torch.FloatTensor(int(utils.prod(shape))).uniform_().view(shape), requires_grad=False),
		'stochastic': True
	},

	'get_shape': {
		'in_types': ['stacks'],
		'out_type': 'shape',
		'fn': get_shape,
		'stochastic': False
	},

	'shape_1d': {
		'in_types': ['integer'],
		'out_type': 'shape',
		'fn': lambda *vec: torch.Size(vec),
		'stochastic': False
	},
	
	'shape_2d': {
		'in_types': ['integer', 'integer'],
		'out_type': 'shape',
		'fn': lambda *vec: torch.Size(vec),
		'stochastic': False
	},

	'shape_3d': {
		'in_types': ['integer', 'integer', 'integer'],
		'out_type': 'shape',
		'fn': lambda *vec: torch.Size(vec),
		'stochastic': False
	},

	'shape_4d': {
		'in_types': ['integer', 'integer', 'integer', 'integer'],
		'out_type': 'shape',
		'fn': lambda *vec: torch.Size(vec),
		'stochastic': False
	},

}


Instruction_probabilities = {
	'matmul': 15,
	'add': 15,
	'relu': 10,
	'folded_normal': 15,
	'uniform': 15,
	'get_shape': 2,
	'shape_1d': 0.75,
	'shape_2d': 0.75,
	'shape_3d': 0.75,
	'shape_4d': 0.75,
}
