from typing import List, Tuple, Set
from src.instance import TSPInstance
from src.genome import TwinPathsGenome

class MemeticLocalSearch:
    """
    Handles local optimization (2-opt) tailored for the Minimax TSP problem,
    ensuring compliance with edge-disjoint constraints during improvement.
    """
    
    @staticmethod
    def run_2opt_on_path(path: List[int], other_path_edges: Set[Tuple[int, int]], instance: TSPInstance) -> List[int]:
        """
        Performs a standard 2-opt local search on a single path, but REJECTS any move
        that creates an edge existing in the other path.
        Returns the best improved valid path found.
        """
        best_path = list(path)
        num_cities = instance.num_cities
        improved = True
        
        while improved:
            improved = False
            for i in range(1, num_cities - 1):
                for j in range(i + 1, num_cities):
                    if j - i == 1:
                        continue # No change if reversing a single node
                    
                    # 1. Simulate the 2-opt swap (reverse segment from i to j)
                    new_path = best_path[:]
                    new_path[i:j] = reversed(best_path[i:j])
                    
                    # 2. Extract only the newly formed edges to check validation against the other path
                    # In a 2-opt swap, only the boundary edges change connection:
                    # Old edges: (i-1 -> i) and (j-1 -> j) [if wrapped around properly]
                    # New edges formed: (i-1 -> j-1) and (i -> j)
                    # For safety and ease, we can fetch all edges or just validate the modified boundaries.
                    # Let's check the entire new path's validity quickly:
                    is_valid_move = True
                    n = len(new_path)
                    
                    # Quick edge check
                    for k in range(n):
                        u, v = new_path[k], new_path[(k + 1) % n]
                        edge = (min(u, v), max(u, v))
                        if edge in other_path_edges:
                            is_valid_move = False
                            break
                    
                    if not is_valid_move:
                        continue # Skip this move, it violates the edge-disjoint constraint
                    
                    # 3. Compute cost change
                    # Cost calculation helper (could also be optimized to delta-evaluation, but full is safe)
                    def get_cost(p):
                        return sum(instance.get_distance(p[k], p[(k + 1) % len(p)]) for k in range(len(p)))
                        
                    if get_cost(new_path) < get_cost(best_path):
                        best_path = new_path
                        improved = True
                        break # Restart the search with the improved path (First Improvement strategy)
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
            # Recompute intermediate edges of optimized Path 1
            temp_genome = TwinPathsGenome(optimized_p1, genome.path2, instance)
            # Now optimize Path 2 against the new Path 1 edges
            optimized_p2 = MemeticLocalSearch.run_2opt_on_path(genome.path2, temp_genome.edges1, instance)
        else:
            # Path 2 is worse, optimize it against fixed Path 1 edges
            optimized_p2 = MemeticLocalSearch.run_2opt_on_path(genome.path2, genome.edges1, instance)
            temp_genome = TwinPathsGenome(genome.path1, optimized_p2, instance)
            optimized_p1 = MemeticLocalSearch.run_2opt_on_path(genome.path1, temp_genome.edges2, instance)
            
        return TwinPathsGenome(optimized_p1, optimized_p2, instance)