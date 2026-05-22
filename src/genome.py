from typing import List, Tuple, Set
from src.instance import TSPInstance
from src.initialization import PopulationInitializer

class TwinPathsGenome:
    """
    Represents an individual solution in the population, containing two Hamiltonian paths.
    Tracks costs, edge overlap, and evaluates fitness based on the assignment rules.
    """
    def __init__(self, path1: List[int], path2: List[int], instance: TSPInstance):
        self.path1 = path1
        self.path2 = path2
        self.instance = instance
        
        # 1. Compute costs for both paths
        self.c1 = self._calculate_path_cost(self.path1)
        self.c2 = self._calculate_path_cost(self.path2)
        
        # 2. Extract edge sets and find any overlap
        self.edges1 = self._get_edges(self.path1)
        self.edges2 = self._get_edges(self.path2)
        self.overlapping_edges = self.edges1.intersection(self.edges2)

    def _calculate_path_cost(self, path: List[int]) -> float:
        """Calculates the total Euclidean distance of a closed Hamiltonian path."""
        cost = 0.0
        n = len(path)
        for i in range(n):
            cost += self.instance.get_distance(path[i], path[(i + 1) % n])
        return cost

    def _get_edges(self, path: List[int]) -> Set[Tuple[int, int]]:
        """Extracts all edges from a path as sorted pairs (undirected graph)."""
        edges = set()
        n = len(path)
        for i in range(n):
            u = path[i]
            v = path[(i + 1) % n]
            # Always keep smaller node index first to match undirected edges
            edges.add((min(u, v), max(u, v)))
        return edges

    @classmethod
    def create_random(cls, instance: TSPInstance) -> 'TwinPathsGenome':
        """Factory method to generate a smart initialized individual using our PopulationInitializer."""
        p1 = PopulationInitializer.generate_random_permutation(instance.num_cities)
        
        # Extract path 1 edges to pass them as forbidden to path 2
        dummy_genome = cls(p1, p1, instance) # Temporary step to easily grab edges
        edges1 = dummy_genome.edges1
        
        p2 = PopulationInitializer.generate_disjoint_path(instance, edges1)
        return cls(p1, p2, instance)

    @property
    def fitness(self) -> Tuple[float, float, float, float]:
        """
        Calculates the lexicographical fitness score for the Minimax TSP problem.
        Python compares tuples left-to-right, perfectly matching the assignment requirements:
        1. Penalty for overlapping edges (Must be 0 for a valid solution).
        2. Minimax Target: The cost of the longest path (max(c1, c2)).
        3. Sum of Paths: Total cost of both paths (c1 + c2) - acts as 1st tie-breaker.
        4. Imbalance: Difference between path costs (|c1 - c2|) - acts as 2nd tie-breaker.
        """
        penalty = len(self.overlapping_edges) * 10000.0
        max_cost = max(self.c1, self.c2)
        sum_cost = self.c1 + self.c2
        imbalance = abs(self.c1 - self.c2)
        
        return (penalty, max_cost, sum_cost, imbalance)

    def is_valid(self) -> bool:
        """Returns True if there is no edge overlap between the two paths."""
        return len(self.overlapping_edges) == 0

    def get_lexicographical_scores(self) -> Tuple[float, float, float]:
        """
        Returns the evaluation metrics requested in Chapter 2, Part 1-f:
        1) max(c(T1), c(T2))
        2) c(T1) + c(T2)
        3) |c(T1) - c(T2)|
        """
        return max(self.c1, self.c2), (self.c1 + self.c2), abs(self.c1 - self.c2)