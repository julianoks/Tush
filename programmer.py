import instructions, utils
import random

class program_generator(object):
	PARAMS = {
		'default_type_probs': {'exec': 5, 'integer': 2, 'blueprint': 3},
		'instructions': instructions.Instructions,
		'instruction_probs': instructions.Instruction_probabilities,
		'integer_generator': lambda : 2**random.randint(0,6),
		'blueprint_size': lambda : 28 #random.randint(15,25)
	}

	def __init__(self, type_probs=None):
		''' Setup probabilities for selecting from each stack type, and instructions for exec stack. '''
		if type_probs is None: type_probs = self.PARAMS['default_type_probs']
		self.type_sampler = utils.random_sampler(type_probs)
		self.stochastic_instruction_sampler = utils.random_sampler(self.PARAMS['instruction_probs'])
		determ_probs = dict(filter(lambda x: not self.PARAMS['instructions'][x[0]]['stochastic'], self.PARAMS['instruction_probs'].items()))
		self.deterministic_instruction_sampler = utils.random_sampler(determ_probs)

	def generate_block(self, in_blueprint):
		stack = self.type_sampler.pick_event()
		if stack == 'exec': 
			if in_blueprint: return ['exec', self.stochastic_instruction_sampler.pick_event()]
			else: return ['exec', self.deterministic_instruction_sampler.pick_event()]
		elif stack == 'integer':
			return ['integer', self.PARAMS['integer_generator']()]
		elif stack == 'blueprint':
			return ['blueprint', None]
		
	def generate_blocks(self, n, use_blueprints):
		blocks = []
		while len(blocks) < n:
			block = self.generate_block(not use_blueprints)
			if use_blueprints or block[0] != 'blueprint':
				blocks.append(block)
		return blocks

	def generate_program(self, n):
		blocks = self.generate_blocks(n, True)
		for i, (stack, val) in enumerate(blocks):
			if stack == 'blueprint':
				blocks[i][1] = self.generate_blocks(self.PARAMS['blueprint_size'](), False)
		return blocks
