import tush
import torch

program = tush.program_generator().generate_program(500)
ind = tush.Tush(program)
ind.populate_input([['tensor', torch.ones([2,16])]])

print(ind.execute_program().get_tensor_out([10]))
