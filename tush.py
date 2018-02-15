import torch
import instructions
from instructions import Instructions
import utils
import random

class program_generator(object):
	def __init__(self, probs=None):
		''' Setup probabilities for selecting from each stack type, and instructions for exec stack. '''
		default_type_probs = {'exec': 5, 'integer': 2, 'blueprint': 1}
		if probs is None: probs = default_type_probs
		self.type_sampler = utils.random_sampler(probs)
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
			if use_blueprints or block[0] != 'blueprint': blocks.append(block)
		return blocks

	def generate_program(self, n):
		blocks = self.generate_blocks(n, True)
		for i, (typ, val) in enumerate(blocks):
			if typ == 'blueprint':
				blocks[i][1] = self.generate_blocks(random.randint(15,25), False)
		return blocks





class Tush(object):
	def __init__(self, program):
		self.stack_types = ['exec', 'tensor', 'shape', 'integer']
		self.hyperparams = {'learning_rate': 0.01, 'steps': 500}
		program_stage_one, self.blueprint_vars = self.stage_one(program)
		self.stage_two_stacks = self.stage_two(program_stage_one)

	def populate_input(self, stacks, input_instructions):
		for stack, instr in input_instructions:
			if stack == 'tensor' and type(instr) != torch.autograd.variable.Variable:
				stacks[stack].insert(0, torch.autograd.Variable(instr, requires_grad=False))
			else: stacks[stack].insert(0, instr)
		return stacks

	def stage_one(self, orig_program):
		''' Replace blueprints with variables '''
		program = utils.copy_w_vars(orig_program)
		variables = []
		for i, (stack, instr) in reversed(list(enumerate(program))):
			if stack != 'blueprint': continue
			out = Tush(instr)
			out = out.execute_program(out.stage_two_stacks)['tensor']
			if not out: program.pop(i)
			else:
				program[i] = ['tensor', torch.autograd.Variable(out[0].data, requires_grad=True)]
				variables.append(program[i][1])
		return program, variables

	def stage_two(self, program_stage_one):
		''' Populate stacks '''
		stacks = {stack: [] for stack in self.stack_types}
		for stack, instr in program_stage_one:
			stacks[stack].append(instr)
		return stacks

	def execute_step(self, stacks):
		instr = stacks['exec'].pop(0)
		inputs = []
		in_types = Instructions[instr]['in_types']
		def put_back():
			for i,t in reversed(list(zip(inputs, in_types[:len(inputs)]))):
				if t != 'stacks': stacks[t].insert(0,i)
		for needed in in_types:
			if needed == 'stacks':
				inputs.append(stacks)
				continue
			if not stacks[needed]:
				put_back()
				return stacks
			inputs.append(stacks[needed].pop(0))
		try:
			out = Instructions[instr]['fn'](*inputs)
			stacks[Instructions[instr]['out_type']].append(out)
		except (ValueError, RuntimeError):
			put_back()
		return stacks

	def execute_program(self, stacks):
		while stacks['exec']: stacks = self.execute_step(stacks)
		return stacks

	def get_tensor_out(self, stacks, shape):
		if shape is None: return stacks # for if loss fn wants to retrieve values directly
		required = int(utils.prod(shape))
		for tensor in reversed(stacks['tensor']):
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
		y_hats = self.get_outputs(x_yshape_pairs)
		loss = sum([loss_fn(y_hat, y) for yhat, y in zip(y_hats, ys)])
		loss /= len(ys)
		return loss
