import random
from typing import List, Tuple, Set
from src.instance import TSPInstance

class PopulationInitializer:
    """
    Class responsible for generating initial solutions for the population.
    Its primary goal is to produce two Hamiltonian paths with minimal edge overlap.
    """
    @staticmethod
    def generate_random_permutation(num_cities: int) -> List[int]:
        """Generates a valid random path (a permutation of the cities)."""
        path = list(range(num_cities))
        random.shuffle(path)
        return path

    @staticmethod
    def generate_disjoint_path(instance: TSPInstance, forbidden_edges: Set[Tuple[int, int]]) -> List[int]:
        """
        Builds a second path using a greedy-probabilistic approach (Nearest Neighbor combined with roulette wheel selection),
        while heavily penalizing edges present in forbidden_edges (the edges of the first path).
        """
        num_cities = instance.num_cities
        unvisited = set(range(num_cities))
        
        # Choose a random starting city
        current_city = random.choice(list(unvisited))
        path = [current_city]
        unvisited.remove(current_city)
        
        while unvisited:
            candidates = list(unvisited)
            weights = []
            
            for next_city in candidates:
                # Define the edge as an unordered pair (smaller index always first)
                edge = (min(current_city, next_city), max(current_city, next_city))
                base_dist = instance.get_distance(current_city, next_city)
                
                # If the edge is forbidden (overlaps with path 1), add a severe artificial penalty
                penalty = 10000.0 if edge in forbidden_edges else 0.0
                total_cost = base_dist + penalty
                
                # Convert the cost to a positive weight for selection (smaller cost = higher weight)
                # Add a small epsilon to prevent division by zero
                weight = 1.0 / (total_cost + 1e-6)
                weights.append(weight)
            
            # Select the next neighbor using weighted roulette wheel selection
            total_w = sum(weights)
            if total_w == 0:
                next_city = random.choice(candidates)
            else:
                r = random.uniform(0, total_w)
                current_sum = 0
                for idx, w in enumerate(weights):
                    current_sum += w
                    if current_sum >= r:
                        next_city = candidates[idx]
                        break
            
            path.append(next_city)
            unvisited.remove(next_city)
            current_city = next_city
            
        return path