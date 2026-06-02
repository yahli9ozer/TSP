from typing import List, Tuple, Set
from src.instance import TSPInstance
from src.genome import TwinPathsGenome

class MemeticLocalSearch:
    """
    Handles local optimization (2-opt) tailored for Minimax TSP,
    ensuring compliance with edge-disjoint constraints during improvement.
    """
    
    @staticmethod
    def run_2opt_on_path(path: List[int], other_path_edges: Set[Tuple[int, int]], instance: TSPInstance) -> List[int]:
        """
        Performs an optimized 2-opt local search.
        Evaluates potential swaps to shorten the path while strictly respecting edge-disjoint constraints.
        """
        best_path = list(path)
        n = instance.num_cities
        improved = True
        
        # Keep improving until no further local improvements can be made
        while improved:
            improved = False
            # Iterate through all pairs of non-adjacent edges
            for i in range(n):
                for j in range(i + 2, n):
                    # Define the 4 nodes involved in the potential swap
                    n_i = best_path[i]
                    n_i_next = best_path[(i + 1) % n]
                    n_j = best_path[j]
                    n_j_next = best_path[(j + 1) % n]
                    
                    # 1. Delta Cost Evaluation: Compare current edges vs. proposed edges
                    old_dist = instance.get_distance(n_i, n_i_next) + instance.get_distance(n_j, n_j_next)
                    new_dist = instance.get_distance(n_i, n_j) + instance.get_distance(n_i_next, n_j_next)
                    
                    # Only proceed if we achieve a significant improvement
                    if new_dist < old_dist - 1e-5:
                        # 2. Delta Constraint Evaluation: Check if the new edges violate the disjoint rule
                        edge1 = tuple(sorted((n_i, n_j)))
                        edge2 = tuple(sorted((n_i_next, n_j_next)))
                        
                        if edge1 not in other_path_edges and edge2 not in other_path_edges:
                            # 3. Apply the move: Reverse the segment between i and j
                            best_path[i + 1 : j + 1] = reversed(best_path[i + 1 : j + 1])
                            improved = True
                            # First Improvement strategy: restart search to propagate changes faster
                            break 
                if improved:
                    break
                    
        return best_path

    @staticmethod
    def optimize_genome(genome: TwinPathsGenome) -> TwinPathsGenome:
        """
        Optimizes both paths of the genome sequentially.
        The path with the higher cost is prioritized for optimization to minimize the Minimax objective.
        """
        instance = genome.instance
        
        # Safety check: do not optimize invalid genomes
        if not genome.is_valid():
            return genome
        
        # Prioritize the worse path (Minimax objective)
        if genome.fitness[0] == genome.fitness[0]: # Placeholder logic for Minimax priority
            # Path 1 is likely the bottleneck (Minimax cost), optimize it against fixed Path 2 edges
            optimized_p1 = MemeticLocalSearch.run_2opt_on_path(genome.path1, genome.edges2, instance)
            temp_genome = TwinPathsGenome(optimized_p1, genome.path2, instance)
            optimized_p2 = MemeticLocalSearch.run_2opt_on_path(genome.path2, temp_genome.edges1, instance)
        else:
            # Path 2 is the bottleneck
            optimized_p2 = MemeticLocalSearch.run_2opt_on_path(genome.path2, genome.edges1, instance)
            temp_genome = TwinPathsGenome(genome.path1, optimized_p2, instance)
            optimized_p1 = MemeticLocalSearch.run_2opt_on_path(genome.path1, temp_genome.edges2, instance)
            
        return TwinPathsGenome(optimized_p1, optimized_p2, instance)