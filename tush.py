import torch
import instructions
from instructions import Instructions
import utils
import random

class program_generator(object):
	def __init__(self, type_probs=None):
		''' Setup probabilities for selecting from each stack type, and instructions for exec stack. '''
		default_type_probs = {'exec': 5, 'integer': 2, 'blueprint': 3}
		if type_probs is None: type_probs = default_type_probs
		self.type_sampler = utils.random_sampler(type_probs)
		self.instruction_sampler = utils.random_sampler(instructions.Instruction_probabilities)

	def generate_block(self):
		stack = self.type_sampler.pick_event()
		if stack == 'exec': 
			return ['exec', self.instruction_sampler.pick_event()]
		elif stack == 'integer':
			return ['integer', 2**random.randint(0,6)]
		elif stack == 'blueprint':
			return ['blueprint', None]
		
	def generate_blocks(self, n, use_blueprints):
		blocks = []
		while len(blocks) < n:
			block = self.generate_block()
			if use_blueprints or block[0] != 'blueprint':
				blocks.append(block)
		return blocks

	def generate_program(self, n):
		blocks = self.generate_blocks(n, True)
		for i, (stack, val) in enumerate(blocks):
			if stack == 'blueprint':
				blocks[i][1] = self.generate_blocks(random.randint(15,25), False)
		return blocks




class Tush(object):
	def __init__(self, program):
		self.stack_types = ['exec', 'tensor', 'shape', 'integer']
		program_stage_one, self.blueprint_vars = self.stage_one(program)
		self.stage_two_stacks = self.stage_two(program_stage_one)
		self.reg_strength = 0.001
		self.out_dependent = True # whether we can output an item that independent of input

	def populate_input(self, stacks, input_instructions):
		for stack, instr in input_instructions:
			if stack == 'tensor' and type(instr) != torch.autograd.variable.Variable:
				item = torch.autograd.Variable(instr, requires_grad=False)
			else: item = instr
			stacks[stack].insert(0, {'val': item, 'input_dep': True})
		return stacks

	def stage_one(self, orig_program):
		''' Replace blueprints with variables '''
		program = []
		variables = []
		for stack, instr in orig_program:
			if stack != 'blueprint':
				program.append([stack,instr])
			else:
				out = Tush(instr)
				out.out_dependent = False
				out = out.execute_program(out.stage_two_stacks)['tensor']
				if not out: continue
				else:
					variables.append(torch.autograd.Variable(out[0]['val'].data, requires_grad=True))
					program.append(['tensor', variables[-1]])
		return program, variables

	def stage_two(self, program_stage_one):
		''' Populate stacks, mark as being independent of input '''
		stacks = {stack: [] for stack in self.stack_types}
		for stack, instr in program_stage_one:
			stacks[stack].append({'val': instr, 'input_dep': False})
		return stacks

	def execute_step(self, stacks):
		exec_item = stacks['exec'].pop(0)
		instr, instr_dep = exec_item['val'], exec_item['input_dep']
		inputs = []
		in_types = Instructions[instr]['in_types']
		def put_back():
			for i,t in reversed(list(zip(inputs, in_types[:len(inputs)]))):
				if t != 'stacks': stacks[t].insert(0,i)
		for needed in in_types:
			if needed == 'stacks':
				inputs.append({'val': stacks, 'input_dep': False}) # stacks objects is nominally independent of input
				continue
			if not stacks[needed]:
				put_back()
				return stacks
			inputs.append(stacks[needed].pop(0))
		try:
			out = Instructions[instr]['fn'](*[i['val'] for i in inputs])
			dependent = any([i['input_dep'] for i in inputs]) or instr_dep
			stacks[Instructions[instr]['out_type']].append({'val': out, 'input_dep': dependent})
		except (ValueError, RuntimeError):
			put_back()
		return stacks

	def execute_program(self, stacks):
		while stacks['exec']: stacks = self.execute_step(stacks)
		return stacks

	def get_tensor_out(self, stacks, shape):
		if shape is None: return stacks # for if loss fn wants to retrieve values directly
		required = int(utils.prod(shape))
		for item in reversed(stacks['tensor']):
			tensor, dependent = item['val'], item['input_dep']
			if self.out_dependent and not dependent: continue
			if int(utils.prod(tensor.shape)) < required: continue
			return tensor.view(-1)[:required].view(shape)
		return None

	def get_output(self, input_instructions, output_shape):
		stacks = utils.copy_w_vars(self.stage_two_stacks)
		stacks = self.populate_input(stacks, input_instructions)
		stacks = self.execute_program(stacks)
		return self.get_tensor_out(stacks, output_shape)

	def get_outputs(self, x_yshape_pairs):
		return [self.get_output(*pair) for pair in x_yshape_pairs]

	def get_loss(self, x_yshape_pairs, ys, loss_fn):
		''' average data loss '''
		y_hats = self.get_outputs(x_yshape_pairs)
		loss = sum([loss_fn(y_hat, y) for y_hat, y in zip(y_hats, ys)])
		loss /= len(ys)
		return loss		

	def optimize(self, train_batch, loss_fn, validation_batch=None, lr=0.05):
		'''
		args:
			train_batch - a list of (x,y) pairs such that
							x is a pair of (program_input, output_shape) and
							y is list of target, such that
								program_input is a list of instructions
								output_shape gives the shape to extract from the tensor stack
			loss_fn - loss function with two inputs: (predicted, target) -> data_loss
			validation_batch - optional dataset in the same format as batches, on which to run and return the validation loss
		returns:
			if validation_batch is used, then returns data loss on the validation data.
		note:
			there are side effects, namely that the blueprint variables will be optimized
		'''
		optimizer = torch.optim.SGD(self.blueprint_vars, lr=lr, momentum=0.9)
		for i, (xs, ys) in enumerate(train_batch):
			optimizer.zero_grad()
			loss = self.get_loss(xs, ys, loss_fn) # data loss
			if 0==i%250: print("\nStep:", i, "\nData Loss:", loss)
			loss += self.reg_strength * sum([(x**2).view(-1).sum() for x in self.blueprint_vars]) # reg. loss
			loss.backward()
			optimizer.step()
		if validation_batch:
			return sum([self.get_loss(xs, ys, loss_fn) for xs,ys in validation_batch]) / len(validation_batch)
