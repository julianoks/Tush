import torch
import instructions
from instructions import Instructions
import utils
import random, copy

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
		stacks = ['exec', 'tensor', 'shape', 'integer']
		self.stacks = {stack: [] for stack in stacks}
		self.hyperparams = {'learning_rate': 0.01, 'steps': 500}
		program_stage_one, self.blueprint_vars = self.stage_one(program)
		self.stage_two(program_stage_one)

	def populate_input(self, program):
		for stack, instr in program:
			if stack == 'tensor' and type(instr) != torch.autograd.variable.Variable:
				self.stacks[stack].insert(0, torch.autograd.Variable(instr, requires_grad=False))
			else: self.stacks[stack].insert(0, instr)

	def stage_one(self, orig_program):
		''' Replace blueprints with variables '''
		program = copy.deepcopy(orig_program)
		variables = []
		for i, (stack, instr) in reversed(list(enumerate(program))):
			if stack != 'blueprint': continue
			out = Tush(instr).execute_program().stacks['tensor']
			if not out: program.pop(i)
			else:
				program[i] = ['tensor', torch.autograd.Variable(out[0].data, requires_grad=True)]
				variables.append(program[i][1])
		return program, variables

	def stage_two(self, program_stage_one):
		''' Populate stacks '''
		for stack, instr in program_stage_one:
			self.stacks[stack].append(instr)

	def execute_step(self):
		instr = self.stacks['exec'].pop(0)
		inputs = []
		in_types = Instructions[instr]['in_types']
		def put_back():
			for i,t in reversed(list(zip(inputs, in_types[:len(inputs)]))):
				if t != 'self': self.stacks[t].insert(0,i)
		for needed in in_types:
			if needed == 'self':
				inputs.append(self)
				continue
			if not self.stacks[needed]:
				put_back()
				return
			inputs.append(self.stacks[needed].pop(0))
		try:
			out = Instructions[instr]['fn'](*inputs)
			self.stacks[Instructions[instr]['out_type']].append(out)
		except (ValueError, RuntimeError): put_back()

	def execute_program(self):
		while self.stacks['exec']: self.execute_step()
		return self

	def get_tensor_out(self, shape):
		required = int(utils.prod(shape))
		for tensor in reversed(self.stacks['tensor']):
			if int(utils.prod(tensor.shape)) < required: continue
			return tensor.view(-1)[:required].view(shape)
		return None
