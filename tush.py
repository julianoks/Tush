import torch
from instructions import Instructions
import utils

class Tush(object):
	def __init__(self, program):
		self.stack_types = ['exec', 'tensor', 'shape', 'integer']
		self.reg_strength = 0.0001
		self.constraint = {'input': True, 'variable': True}
		self.stage_one_stacks, self.blueprint_vars = self.stage_one(program)

	def populate_input(self, stacks, input_instructions):
		for stack, instr in input_instructions:
			if stack == 'tensor' and type(instr) != torch.autograd.variable.Variable:
				item = torch.autograd.Variable(instr, requires_grad=True) # does not actually require grad, but turned on for debugging
			else: item = instr
			stacks[stack].insert(0, {'val': item, 'input_dep': True, 'variable_dep': True})
		return stacks

	def stage_one(self, orig_program):
		''' Replace blueprints with variables '''
		program = []
		variables = []
		for stack, instr in orig_program:
			if stack != 'blueprint':
				program.append([stack,{'val': instr, 'input_dep': False, 'variable_dep': False}])
			else:
				out = Tush(instr)
				out.constraint['input'] = False
				out.constraint['variable'] = False
				out = out.execute_program(out.stage_one_stacks)['tensor']
				if not out: continue
				else:
					variables.append(torch.autograd.Variable(out[0]['val'].data, requires_grad=True))
					program.append(['tensor', {'val': variables[-1], 'input_dep': False, 'variable_dep': True}])
		stacks = {stack: [] for stack in self.stack_types}
		for stack, item in program:
			stacks[stack].append(item)
		return stacks, variables

	def execute_step(self, stacks):
		exec_item = stacks['exec'].pop(0)
		instr, instr_inp_dep, instr_var_dep = exec_item['val'], exec_item['input_dep'], exec_item['variable_dep']
		inputs = []
		in_types = Instructions[instr]['in_types']
		def put_back():
			for i,t in reversed(list(zip(inputs, in_types[:len(inputs)]))):
				if t != 'stacks': stacks[t].insert(0,i)
		for needed in in_types:
			if needed == 'stacks':
				inputs.append({'val': stacks, 'input_dep': False, 'variable_dep': False}) # stacks object is nominally independent
				continue
			if not stacks[needed]:
				put_back()
				return stacks
			inputs.append(stacks[needed].pop(0))
		try:
			out = Instructions[instr]['fn'](*[i['val'] for i in inputs])
			inp_dep = any([i['input_dep'] for i in inputs]) or instr_inp_dep
			var_dep = any([i['variable_dep'] for i in inputs]) or instr_var_dep
			stacks[Instructions[instr]['out_type']].append({'val': out, 'input_dep': inp_dep, 'variable_dep': var_dep})
		except (ValueError, RuntimeError):
			put_back()
		return stacks

	def execute_program(self, stacks):
		while stacks['exec']: stacks = self.execute_step(stacks)
		return stacks

	def get_tensor_out(self, stacks, shape, con_inp=None, con_var=None):
		if con_inp is None: con_inp = self.constraint['input']
		if con_var is None: con_var = self.constraint['variable']
		for item in reversed(stacks['tensor']):
			tensor, inp_dep, var_dep = item['val'], item['input_dep'], item['variable_dep']
			if con_inp and not inp_dep: continue
			if con_var and not var_dep: continue
			if len(shape) > len(tensor.shape): continue
			if not all([a<=b for a,b in zip(shape, tensor.shape[:len(shape)])]): continue
			out = tensor # drop last indices, to have same num dims as shape
			for i, l in enumerate(shape): out = out.narrow(i,0,l)
			for i in range(len(shape), len(tensor.shape)): out = out.narrow(i,0,1)
			return out
		if con_inp or con_var:
			return self.get_tensor_out(stacks, shape, con_inp=False, con_var=False)
		return torch.autograd.Variable(torch.ones(shape)) # if no valid output, return ones, so as to maximize entropy

	def get_output(self, input_instructions, output_shape):
		stacks = utils.copy_w_vars(self.stage_one_stacks)
		stacks = self.populate_input(stacks, input_instructions)
		stacks = self.execute_program(stacks)
		return self.get_tensor_out(stacks, output_shape)

	def get_outputs(self, x_yshape_pairs):
		return [self.get_output(x,yshape) for x,yshape in x_yshape_pairs]

	def get_loss(self, x_yshape_pairs, ys, loss_fn):
		''' average data loss '''
		y_hats = self.get_outputs(x_yshape_pairs)
		loss = sum([loss_fn(y_hat, y) for y_hat, y in zip(y_hats, ys)])
		loss /= len(ys)
		return loss

	def optimize(self, train_batch, loss_fn, validation_batch=None):
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
		if self.blueprint_vars:
			optimizer = torch.optim.Adam(self.blueprint_vars, lr=0.001)
			for i, (xs, ys) in enumerate(train_batch):
				optimizer.zero_grad()
				loss = self.get_loss(xs, ys, loss_fn) # data loss
				if 0==i%500: print("\nStep:", i, "\nData Loss:", loss); print(self.blueprint_vars)
				loss += self.reg_strength * sum([(x**2).view(-1).sum() for x in self.blueprint_vars]) # reg. loss
				loss.backward()
				optimizer.step()
		else: print("No variables to optimize! No optimization took place!")
		if validation_batch:
			return sum([self.get_loss(xs, ys, loss_fn) for xs,ys in validation_batch]) / len(validation_batch)
