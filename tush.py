import torch
from instructions import Instructions
import utils
import numpy as np

class Tush(object):
	def __init__(self, program):
		self.print_every = 10
		self.stack_types = ['exec', 'tensor', 'shape', 'integer', 'bool']
		self.reg_strength = 0.0001
		self.constraint = {'input': True, 'variable': True}
		self.stage_one_stacks, self.blueprint_vars = self.stage_one(program)

	def stage_one(self, orig_program):
		''' Replace blueprints with variables '''
		program = []
		variables = []
		for stack, instr in orig_program:
			if stack != 'blueprint':
				program.append({'val': instr, 'input_dep': False, 'variable_dep': False, 'stack': stack})
			else:
				out = Tush(instr)
				out.constraint['input'] = False
				out.constraint['variable'] = False
				out = out.execute_program(out.stage_one_stacks)['tensor']
				if not out: continue
				else:
					variables.append(torch.autograd.Variable(out[0]['val'].data, requires_grad=True))
					program.append({'val': variables[-1], 'input_dep': False, 'variable_dep': True, 'stack': 'tensor'})
		stacks = {stack: [] for stack in self.stack_types}
		for item in program:
			stacks['exec'].append(item)
		return stacks, variables

	def execute_step(self, stacks):
		this_item = stacks['exec'].pop()
		stack_name = this_item['stack']
		if stack_name != 'exec':
			if stack_name == 'code': assert True, "This hasn't been implemented yet"
			else: stacks[stack_name].append(this_item)
			return stacks
		instr, instr_inp_dep, instr_var_dep = this_item['val'], this_item['input_dep'], this_item['variable_dep']
		inputs = []
		in_types = Instructions[instr]['in_types']
		def put_back():
			for i in reversed(inputs):
				if i['stack'] is None: continue # case where argument is 'stacks'
				stacks[i['stack']].append(i)
		for needed in in_types:
			if needed == 'stacks':
				inputs.append({'val': stacks, 'input_dep': False, 'variable_dep': False, 'stack': None}) # stacks object is nominally independent
				continue
			if not stacks[needed]:
				put_back()
				return stacks
			inputs.append(stacks[needed].pop())
		try:
			out = Instructions[instr]['fn'](*[i['val'] for i in inputs])
			inp_dep = any([i['input_dep'] for i in inputs]) or instr_inp_dep
			var_dep = any([i['variable_dep'] for i in inputs]) or instr_var_dep
			out_stack = Instructions[instr]['out_type']
			stacks[out_stack].append({'val': out, 'stack': out_stack, 'input_dep': inp_dep, 'variable_dep': var_dep})
		except (ValueError, RuntimeError):
			put_back()
		return stacks

	def populate_input(self, stacks, input_instructions):
		for stack, instr in input_instructions:
			if stack == 'tensor' and type(instr) != torch.autograd.variable.Variable:
				val = torch.autograd.Variable(torch.Tensor(instr), requires_grad=True) # does not actually require grad, but turned on for debugging
			else: val = instr
			stacks['exec'].append({'val': val, 'input_dep': True, 'variable_dep': False, 'stack': stack})
		return stacks

	def execute_program(self, stacks):
		tensor_counter = 0
		general_counter = 0
		while stacks['exec']:
			if stacks['exec'][-1]['stack'] == 'exec':
				instruction = Instructions[stacks['exec'][-1]['val']]
				if 'tensor' in instruction['in_types']:
					tensor_counter += 1
				general_counter += 1
			stacks = self.execute_step(stacks)
		return stacks

	def get_tensor_out(self, stacks, shape, con_inp=None, con_var=None):
		if con_inp is None: con_inp = self.constraint['input']
		if con_var is None: con_var = self.constraint['variable']
		for item in reversed(stacks['tensor']):
			orig_tensor, inp_dep, var_dep = item['val'], item['input_dep'], item['variable_dep']
			if con_inp and not inp_dep: continue
			if con_var and not var_dep: continue
			if len(shape) > len(orig_tensor.shape): continue
			tensor = orig_tensor.permute([x[0] for x in sorted(zip(range(len(orig_tensor.shape)), orig_tensor.shape), key=lambda x:-x[1])])
			if not all([a<=b for a,b in zip(shape, tensor.shape[:len(shape)])]): continue
			out = tensor # drop last indices, to have same num dims as shape
			for i, l in enumerate(shape): out = out.narrow(i,0,l)
			for i in range(len(shape), len(tensor.shape)): out = out.narrow(i,0,1)
			return out.view(*shape)
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

	def stage_two(self, train_batch, loss_fn, validation_batch=None):
		'''
		Stage two (optimization)
		args:
			train_batch - a list of (x,y) pairs such that
							x is a pair of (program_input, output_shape) and
							y is list of target, such that
								program_input is a list of instructions
								output_shape gives the shape to extract from the tensor stack
			loss_fn - loss function with two inputs: (predicted, target) -> data_loss
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
				if self.print_every and 0==i%self.print_every:
					print("\nStep:", i, "\nData Loss:", loss); print(self.blueprint_vars)
				loss += self.reg_strength * sum([(x**2).view(-1).sum() for x in self.blueprint_vars]) # reg. loss
				loss.backward()
				optimizer.step()
		else: print("No variables to optimize! No optimization took place!")

	def stage_three(self, validation_batch, loss_fn=None, classification=True):
		results = {}
		yhats = [self.get_outputs(xs) for xs,_ in validation_batch]
		ys = [y for _,y in validation_batch]
		if classification:
			accuracy = [np.array([yh.data.tolist() for yh in yhat]).argmax(1)==y for yhat, y in zip(yhats, ys)]
			accuracy = np.mean(accuracy)
			results['accuracy'] = accuracy
		if loss_fn:
			results['loss'] = float(np.mean([[loss_fn(syh,sy) for syh,sy in zip(yhat,y)] for yhat,y in zip(yhats, ys)]))
		return results
