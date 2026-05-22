import random
from typing import List, Tuple, Set
from src.instance import TSPInstance
from src.genome import TwinPathsGenome

class GeneticOperators:
    """
    Handles reproduction (Crossover), variation (Mutation), and constraint satisfaction (Repair)
    for the Minimax Twin-Paths TSP problem.
    """
    
    # ---------------- CROSSOVER OPERATORS ----------------

    @staticmethod
    def order_crossover(p1: List[int], p2: List[int]) -> List[int]:
        """ OX (Ordered Crossover) - Preserves relative order """
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[start:end] = p1[start:end]
        
        p2_filtered = [x for x in p2 if x not in child]
        idx = 0
        for i in range(size):
            if child[i] == -1:
                child[i] = p2_filtered[idx]
                idx += 1
        return child

    @staticmethod
    def pmx_crossover(p1: List[int], p2: List[int]) -> List[int]:
        """ PMX (Partially Matched Crossover) - Preserves absolute positions via mapping """
        size = len(p1)
        start, end = sorted(random.sample(range(size), 2))
        child = [-1] * size
        
        # 1. Copy segment from parent 1
        child[start:end] = p1[start:end]
        
        # 2. Map elements from parent 2 that are not in the copied segment
        for i in range(start, end):
            if p2[i] not in child:
                val = p2[i]
                curr_idx = i
                # Follow the mapping cycle until we find an empty spot
                while start <= curr_idx < end:
                    mapped_val = p1[curr_idx]
                    curr_idx = p2.index(mapped_val)
                child[curr_idx] = val
                
        # 3. Fill the remaining empty spots with elements from parent 2
        for i in range(size):
            if child[i] == -1:
                child[i] = p2[i]
                
        return child

    # ---------------- MUTATION OPERATORS ----------------

    @staticmethod
    def swap_mutation(path: List[int], mutation_rate: float = 0.1) -> List[int]:
        """ Simple Swap Mutation - Exchanges two random cities """
        mutated = path[:]
        if random.random() < mutation_rate:
            idx1, idx2 = random.sample(range(len(path)), 2)
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
        return mutated

    @staticmethod
    def inversion_mutation(path: List[int], mutation_rate: float = 0.1) -> List[int]:
        """ Inversion Mutation - Reverses a segment of the path (Highly effective for TSP) """
        mutated = path[:]
        if random.random() < mutation_rate:
            start, end = sorted(random.sample(range(len(path) + 1), 2))
            mutated[start:end] = reversed(mutated[start:end])
        return mutated

    # ---------------- REPAIR & BREEDING ----------------

    @staticmethod
    def repair_genome(genome: TwinPathsGenome) -> TwinPathsGenome:
        """ Fixes edge overlaps between Path 1 and Path 2 """
        if genome.is_valid():
            return genome
            
        p1 = genome.path1[:]
        p2 = genome.path2[:]
        instance = genome.instance
        
        max_attempts = 50
        attempts = 0
        current_genome = genome
        
        while not current_genome.is_valid() and attempts < max_attempts:
            overlaps = list(current_genome.overlapping_edges)
            u, v = random.choice(overlaps)
            
            idx_u = p2.index(u)
            swap_idx = random.randint(0, len(p2) - 1)
            p2[idx_u], p2[swap_idx] = p2[swap_idx], p2[idx_u]
            
            current_genome = TwinPathsGenome(p1, p2, instance)
            attempts += 1
            
        return current_genome

    @classmethod
    def reproduce(cls, g1: TwinPathsGenome, g2: TwinPathsGenome, 
                  mutation_rate: float = 0.1,
                  crossover_type: str = 'PMX', 
                  mutation_type: str = 'INVERSION') -> TwinPathsGenome:
        """
        Executes the full breeding cycle using the selected operators.
        Defaults to PMX and INVERSION as they are generally superior for TSP.
        """
        # Select Crossover
        cross_func = cls.pmx_crossover if crossover_type == 'PMX' else cls.order_crossover
        child_p1 = cross_func(g1.path1, g2.path1)
        child_p2 = cross_func(g1.path2, g2.path2)
        
        # Select Mutation
        mut_func = cls.inversion_mutation if mutation_type == 'INVERSION' else cls.swap_mutation
        child_p1 = mut_func(child_p1, mutation_rate)
        child_p2 = mut_func(child_p2, mutation_rate)
        
        child_genome = TwinPathsGenome(child_p1, child_p2, g1.instance)
        return cls.repair_genome(child_genome)