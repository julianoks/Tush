import tush
import torch

program = tush.program_generator().generate_program(50)
ind = tush.Tush(program)
input_in = [['tensor', torch.ones([3,7])]]
input_in_2 = [['tensor', torch.ones([2,16])+1]]

print(ind.get_output(input_in, [10]))
print(ind.get_output(input_in_2, [11]))
#print(ind.stage_two_stacks)
