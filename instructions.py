import torch
import utils

def get_shape(stacks):
	if stacks['tensor']:
		return stacks['tensor'][-1]['val'].shape
	raise ValueError

def duplicate(stacks, stack_name):
	if stacks[stack_name]:
		if not (stack_name=='exec' and stacks[stack_name][-1]['val']=='dup_exec'):
			stacks[stack_name].append(stacks[stack_name][-1])
	raise ValueError

def custom_matmul(x1,x2):
	''' matmul with semantics to dot along as many dimensions as possible '''
	a,b = list(reversed(x1.shape)), list(x2.shape)
	if a[0] != b[0]: raise ValueError
	while len(a)>1 and len(b)>1 and a[1]==b[1]:
		a[0] *= a.pop(1)
		b[0] *= b.pop(1)
	return torch.matmul(x1.view(*reversed(a)), x2.view(*b))

Instructions = {
	
	'matmul': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: custom_matmul(a,b),
		'stochastic': False
	},

	'matmul_backward': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: custom_matmul(b,a),
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

	'conv1d': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.nn.functional.conv1d(a,b,stride=1,padding=0),
		'stochastic': False
	},

	'conv2d': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.nn.functional.conv2d(a,b,stride=1,padding=0),
		'stochastic': False
	},

	'conv3d': {
		'in_types': ['tensor', 'tensor'],
		'out_type': 'tensor',
		'fn': lambda a,b: torch.nn.functional.conv3d(a,b,stride=1,padding=0),
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

	'dup_bool': {
		'in_types': ['stacks'],
		'out_type': 'exec',
		'fn': lambda stacks: duplicate(stacks, 'bool'),
		'stochastic': False
	},

	'bool_and' : {
		'in_types': ['bool', 'bool'],
		'out_type': 'bool',
		'fn': lambda a,b: a and b,
		'stochastic' : False
	},

	'bool_or': {
		'in_types': ['bool', 'bool'],
		'out_type': 'bool',
		'fn': lambda a,b: a or b,
		'stochastic' : False
	},

	'bool_not' : {
		'in_types': ['bool'],
		'out_type': 'bool',
		'fn': lambda a: not a,
		'stochastic' : False
	},

	'bool_xor' : {
		'in_types': ['bool', 'bool'],
		'out_type': 'bool',
		'fn': lambda a,b: a ^ b,
		'stochastic' : False
	},

	'bool_from_int' : {         #Could this be too biased for True?
		'in_types' : ['integer'],
		'out_type' : 'bool',
		'fn': lambda a: bool(a),
		'stochastic' : False
	},

	'if_else' : {
	'in_types' : ['bool','exec', 'exec'],
# 	'out_type' : 'exec',
	'out_type' : 'chk_stack',
	'fn': lambda b, e1, e2 : e1 if b == True else e2,
	'stochastic' : False
	},


}


Instruction_probabilities = {
	'matmul': 10,
	'matmul_backward': 10,
	'add': 15,
	'relu': 10,
	'conv1d': 5,
	'conv2d': 5,
	'conv3d': 5,
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
	'dup_bool': 5,
	'bool_and': 3,
	'bool_or': 3,
	'bool_not': 3,
	'bool_xor': 3,
	'bool_from_int': 1,
	'if_else' : 5
}
