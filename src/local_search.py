from typing import List, Tuple, Set
from src.instance import TSPInstance
from src.genome import TwinPathsGenome

class MemeticLocalSearch:
    """
    Handles local optimization (2-opt) tailored for the Minimax TSP problem,
    ensuring compliance with edge-disjoint constraints during improvement.
    Uses Delta-Evaluation for O(1) performance inside the main loops.
    """
    
    @staticmethod
    def run_2opt_on_path(path: List[int], other_path_edges: Set[Tuple[int, int]], instance: TSPInstance) -> List[int]:
        """
        Performs a highly optimized 2-opt local search using Delta Evaluation.
        Only calculates the cost difference and edge collisions of the 2 swapped edges,
        avoiding full array reconstruction which saves massive amounts of time.
        """
        best_path = list(path)
        num_cities = instance.num_cities
        improved = True
        
        while improved:
            improved = False
            for i in range(1, num_cities - 1):
                # We go up to num_cities + 1 to allow reversing to the very end of the array
                for j in range(i + 1, num_cities + 1): 
                    if j - i == 1:
                        continue # No actual change if reversing a single node
                    
                    # Identify the 4 nodes involved in the 2-opt swap
                    n_prev = best_path[i - 1]
                    n_start = best_path[i]
                    n_end = best_path[j - 1]
                    n_next = best_path[j % num_cities] # Modulo ensures wrap-around
                    
                    # 1. Delta Cost Evaluation O(1)
                    # Current edges cost: (n_prev -> n_start) + (n_end -> n_next)
                    # Proposed edges cost: (n_prev -> n_end) + (n_start -> n_next)
                    old_dist = instance.get_distance(n_prev, n_start) + instance.get_distance(n_end, n_next)
                    new_dist = instance.get_distance(n_prev, n_end) + instance.get_distance(n_start, n_next)
                    
                    # If it doesn't shorten the path (using epsilon for float precision), skip
                    if new_dist >= old_dist - 1e-5:
                        continue
                        
                    # 2. Delta Constraint Evaluation O(1)
                    # Does creating these two new edges violate the disjoint rule?
                    new_edge1 = (min(n_prev, n_end), max(n_prev, n_end))
                    new_edge2 = (min(n_start, n_next), max(n_start, n_next))
                    
                    if new_edge1 in other_path_edges or new_edge2 in other_path_edges:
                        continue # Violates rules, skip
                        
                    # 3. Apply the move (Only if it's both shorter AND legal)
                    best_path[i:j] = reversed(best_path[i:j])
                    improved = True
                    break # Restart search (First Improvement strategy)
                    
                if improved:
                    break
                    
        return best_path

    @staticmethod
    def optimize_genome(genome: TwinPathsGenome) -> TwinPathsGenome:
        """
        Implements Chapter 2, part a & b: Optimize paths independently, 
        starting with the worse (longer) path first to target the minimax cost directly.
        """
        instance = genome.instance
        
        # Decide order: optimize the worse path first
        if genome.c1 >= genome.c2:
            # Path 1 is worse, optimize it against fixed Path 2 edges
            optimized_p1 = MemeticLocalSearch.run_2opt_on_path(genome.path1, genome.edges2, instance)
            temp_genome = TwinPathsGenome(optimized_p1, genome.path2, instance)
            optimized_p2 = MemeticLocalSearch.run_2opt_on_path(genome.path2, temp_genome.edges1, instance)
        else:
            # Path 2 is worse, optimize it against fixed Path 1 edges
            optimized_p2 = MemeticLocalSearch.run_2opt_on_path(genome.path2, genome.edges1, instance)
            temp_genome = TwinPathsGenome(genome.path1, optimized_p2, instance)
            optimized_p1 = MemeticLocalSearch.run_2opt_on_path(genome.path1, temp_genome.edges2, instance)
            
        return TwinPathsGenome(optimized_p1, optimized_p2, instance)