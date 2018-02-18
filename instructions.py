import torch
import utils

def get_shape(stacks):
	if stacks['tensor']:
		return stacks['tensor'][0]['val'].shape
	raise ValueError

def duplicate(stacks, stack_name):
	if stacks[stack_name]:
		stacks[stack_name].append(stacks[stack_name][0])
	raise ValueError

Instructions = {
	
	'matmul': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.matmul(a,b),
		'stochastic': False
	},

	'matmul_backward': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.matmul(b,a),
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

	'dup_exec': {
		'in_types': ['stacks'],
		'out_type': 'exec',
		'fn': lambda stacks: duplicate(stacks, 'exec'),
		'stochastic': False
	},

	'dup_tensor': {
		'in_types': ['stacks'],
		'out_type': 'exec',
		'fn': lambda stacks: duplicate(stacks, 'tensor'),
		'stochastic': False
	},

	'dup_shape': {
		'in_types': ['stacks'],
		'out_type': 'exec',
		'fn': lambda stacks: duplicate(stacks, 'shape'),
		'stochastic': False
	},

	'dup_integer': {
		'in_types': ['stacks'],
		'out_type': 'exec',
		'fn': lambda stacks: duplicate(stacks, 'integer'),
		'stochastic': False
	},

}


Instruction_probabilities = {
	'matmul': 10,
	'matmul_backward': 10,
	'add': 15,
	'relu': 10,
	'folded_normal': 15,
	'uniform': 15,
	'get_shape': 2,
	'shape_1d': 0.75,
	'shape_2d': 0.75,
	'shape_3d': 0.75,
	'shape_4d': 0.75,
	'dup_exec': 5,
	'dup_tensor': 5,
	'dup_shape': 5,
	'dup_integer': 5,
}
