import cv2
import tush
from instructions import Instructions, Instruction_probabilities
import numpy as np
import math
import random
from utils import random_sampler
from programmer import program_generator



class variation(object):
    def __init__(self, mutation_prob=None, program_depth=5, tournament_size=3):
        if mutation_prob==None: mutation_prob={k: 1 for k in range(program_depth)}
        self.parent_program=program_generator().generate_program(program_depth)
        
        
        self.sampler=random_sampler(mutation_prob)
        self.blueprint_size=lambda: random.randint(15,25)
        children=self.uniform_mutation(2*tournament_size, program_depth)
        self.tournament_selection(children)
    
    
    def uniform_mutation(self, children_count, program_depth):
        children=[]
        gene_generator_object=program_generator()
        
        while len(children)<children_count:
            mutator_index=self.sampler.pick_event()
            mutated_child=list(self.parent_program)
            
            mutated_gene=gene_generator_object.generate_block(False)
            
            if (mutated_gene[0]=='blueprint'):
                mutated_gene[1]=gene_generator_object.generate_blocks(self.blueprint_size(), False)
                
            mutated_child[mutator_index]=mutated_gene
            children.append(mutated_child)
        return children
            
                
    def tournament_selection(self, children):
        assert len(children)%2==0, "There must be even number of contenders for tournament selection"
        random.shuffle(children)
        for _ in range(len(children)/2):
            contender_1=children.pop()
            contender_2=children.pop()
        pass
    


        
var=variation()
pass