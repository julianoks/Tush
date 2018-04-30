import cv2
import tush
import random
import torch
import os
import sys
from utils import random_sampler
from programmer import program_generator
from scrubber import data_wrangler
import numpy as np
import logging
from pathos.multiprocessing import Pool
import warnings
warnings.filterwarnings("ignore")

log_file=os.getcwd().rsplit("/",1)[0]+'/logs/tush-'+sys.argv[5]+'.log'

logging.basicConfig(filename=log_file, level=logging.DEBUG,
                            format='%(asctime)s:%(levelname)s:%(message)s\n')
class evolution(object):
    def __init__(self, program_depth=5, population_size=6, mutation_prob=None, par=None):
        assert population_size%2==0, "Genesis population size must be even"
        self.parallel=par
        self.mutation_prob=0.4
        if mutation_prob: self.mutation_prob=mutation_prob
        print "Generating genesis population"
        
        self.genesis_block=[]
        for _ in range(population_size):
            self.genesis_block.append(program_generator().generate_program(program_depth))
            
        print "Genesis population generated"
        
        self.gene_generator_object=program_generator()
        self.blueprint_size=lambda: random.randint(15,25)
    
    def evaluation(self, batches, programs):
        
        def loss_fn(pred, target_idx):
            return - torch.log(torch.nn.functional.softmax(pred)[target_idx])
        
        def train_validate(program, loss_fn, batches):
            ind = tush.Tush(program)
            ind.stage_two(batches['train'], loss_fn)
            results = ind.stage_three(validation_batch=batches['validation'], loss_fn=loss_fn)
            
            return results
        for prog in programs: logging.debug(prog)
        a=lambda x: train_validate(x, loss_fn, batches)['accuracy']
        if self.parallel=='True':
            
           
            p=Pool(len(programs))
            accuracy=p.map(a, programs)

        else:
            accuracy=map(a, programs)            
        return accuracy
        
    def k_way_tournament_selection(self, genome_pool, accuracy, k=2):
        '''
        Returns selected genome and its index
        '''
        if k==1 or len(genome_pool)==1:
            idx=random.randint(0, len(genome_pool)-1)
            return genome_pool[idx], idx
        
        else:
            contender_indices=random.sample(range(len(genome_pool)), k)
            contender_gen_pool=map(lambda x: genome_pool[x], contender_indices)
            contender_acc=map(lambda x: accuracy[x], contender_indices)
            
            return contender_gen_pool[np.argmax(contender_acc)], contender_indices[np.argmax(contender_acc)]
        
           
        
    
    def uniform_mutation(self, children):
        mutation_prob_rate=self.mutation_prob
        for i, child in enumerate(children):
            if random.random()<=mutation_prob_rate:
                
                mutated_gene=self.gene_generator_object.generate_block(False)
                
                if (mutated_gene[0]=='blueprint'):
                    mutated_gene[1]=self.gene_generator_object.generate_blocks(self.blueprint_size(), False)
                mutator_index=random.randint(0,len(child)-1)
                
                children[i][mutator_index]=mutated_gene
        return children
                
                
        assert len(children)%2==0, "There must be even number of contenders for tournament selection"
        random.shuffle(children)
        winners=[]
        for _ in range(len(children)/2):
            contender_1=children.pop()
            contender_2=children.pop()
            
            contender_1_acc=train_validate(contender_1, loss_fn, batches)['accuracy']
            contender_2_acc=train_validate(contender_2, loss_fn, batches)['accuracy']
            
            if contender_1_acc>=contender_2_acc: winners.append(contender_1)
            else: winners.append(contender_2)
        
        return winners
    
    
    
    def two_point_crossover(self, parents):
        assert len(parents)==2, "Only two parents required"
        
        pt_1=random.randint(1,len(parents[0])-1)
        pt_2=random.randint(1,len(parents[0])-1)
        
        while abs(pt_1-pt_2)<2: pt_2=random.randint(0,len(parents[0])-1)
        
        crossover_pt_1=min(pt_1,pt_2)
        crossover_pt_2=max(pt_1,pt_2)
        
        head=[]
        mid=[]
        tail=[]
        
        children=[]
        children.append(parents[0][:crossover_pt_1]+parents[1][crossover_pt_1:crossover_pt_2]+parents[0][crossover_pt_2:])
        children.append(parents[1][:crossover_pt_1]+parents[0][crossover_pt_1:crossover_pt_2]+parents[1][crossover_pt_2:])
        
        return children
        
        
    def start_evolution(self, batches, selection_with_rep=True, cur_gen=0, gen_count=200, programs=None, target_acc=0.9):
        
        
        
        if cur_gen==0: programs=self.genesis_block
        assert programs!=None, "Genome pool unavailable"
        
        accuracy=self.evaluation(batches, programs)
        print "\n\nAccuracy:\t%f" %(max(accuracy))
        if max(accuracy)>=target_acc:
            print "\nTarget achieved at generation %d" %(cur_gen+1)
            return programs[np.argmax(accuracy)], max(accuracy)
        
        if(cur_gen<gen_count):
            
            print "\n#####","Generation %d" %(cur_gen+1), "########"
            children=[]
            if selection_with_rep==False:
                genome_pool=list(programs)
                genome_acc=list(accuracy)
                genome_pool_size=len(genome_pool)
                
                for _ in range(genome_pool_size/2):
                    parents=[]
                    parent_1, idx_1=self.k_way_tournament_selection(genome_pool, genome_acc)
                    parents.append(parent_1)
                    
                    del(genome_pool[idx_1])
                    parent_2, idx_2=self.k_way_tournament_selection(genome_pool, genome_acc)
                    del(genome_pool[idx_2])
                    
                    parents.append(parent_2)
                    
                    children+self.two_point_crossover(parents)
                    
                    
                    
            else:
                genome_pool=list(programs)
                genome_acc=list(accuracy)
                genome_pool_size=len(genome_pool)
                
                for _ in range(genome_pool_size/2):
                    parents=[]
                    parent_1, idx_1=self.k_way_tournament_selection(genome_pool, genome_acc)
                    parents.append(parent_1)
                    
                    parent_2, idx_2=self.k_way_tournament_selection(genome_pool, genome_acc)
                    
                    while idx_1==idx_2: parent_2, idx_2=self.k_way_tournament_selection(genome_pool, genome_acc)
                    
                    parents.append(parent_2)
                
                    children.extend(self.two_point_crossover(parents))
            
            
            mutated_children=self.uniform_mutation(children)
            
            return self.start_evolution(batches, selection_with_rep, cur_gen+1, gen_count, mutated_children, target_acc)
        
        return programs, accuracy
        pass
# a=evolution(mutation_prob=0.6)
# _, acc=a.start_evolution(data_wrangler().get_wine_loaders())
# print "\n\n\nAcc:\t", acc


#    Args: dataset, mutation_prob, population, parallel

if __name__=='__main__':
    if(sys.argv[1]=='wine'):
        _, acc=evolution(mutation_prob=float(sys.argv[2]), population_size=int(sys.argv[3]), par=sys.argv[4]).start_evolution(data_wrangler().get_wine_loaders())
    elif(sys.argv[1]=='mnist'):
        _, acc=evolution(mutation_prob=float(sys.argv[2]), population_size=int(sys.argv[3]), par=sys.argv[4]).start_evolution(data_wrangler().get_mnist_loaders())

            
