import random
from typing import List
from src.instance import TSPInstance
from src.genome import TwinPathsGenome
from src.operators import GeneticOperators
from src.local_search import MemeticLocalSearch

class MinimaxMemeticAlgorithm:
    """
    Main engine for the Minimax Memetic Algorithm.
    Integrates Genetic Algorithms with Local Search (2-opt) to find two edge-disjoint paths.
    """
    def __init__(self, 
                 instance: TSPInstance, 
                 pop_size: int = 50, 
                 generations: int = 100,
                 mutation_rate: float = 0.1,
                 local_search_prob: float = 0.0,
                 elitism_count: int = 2,
                 target_optimum: float = 1.0):  
        """
        Initializes the Memetic Algorithm parameters.
        """
        self.instance = instance
        self.pop_size = pop_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.local_search_prob = local_search_prob
        self.elitism_count = elitism_count
        self.target_optimum = target_optimum
        self.population: List[TwinPathsGenome] = []

    def initialize_population(self):
        """Initializes the population and ensures all starting individuals are strictly valid (no overlaps)."""
        self.population = []
        for _ in range(self.pop_size):
            genome = TwinPathsGenome.create_random(self.instance)
            if not genome.is_valid():
                genome = GeneticOperators.repair_genome(genome)
            self.population.append(genome)

    def tournament_selection(self, k: int = 3) -> TwinPathsGenome:
        """Selects a parent using tournament selection. Relies on Python's native Tuple comparison for lexicographical fitness."""
        contenders = random.sample(self.population, k)
        return min(contenders, key=lambda g: g.fitness)

    def run(self) -> TwinPathsGenome:
        """
        Executes the evolutionary process.
        Prints live progress every 10 generations, including the current ratio to the target optimum.
        """
        self.initialize_population()
        # min() automatically compares fitness Tuples lexicographically
        best_overall = min(self.population, key=lambda g: g.fitness)
        
        print(f"Starting Evolution... Local Search Prob: {self.local_search_prob}")
        
        # Print initial state (Generation 0)
        initial_max_cost = best_overall.fitness[1]
        initial_ratio = initial_max_cost / self.target_optimum
        print(f"Gen   0 | Max Cost: {initial_max_cost:.2f} | Ratio: {initial_ratio:.2f}x | Sum: {best_overall.fitness[2]:.2f}")

        for gen in range(1, self.generations + 1):
            new_population = []
            
            # 1. Elitism: Keep the best individuals automatically
            self.population.sort(key=lambda g: g.fitness)
            new_population.extend(self.population[:self.elitism_count])
            
            # 2. Breeding: Fill the rest of the new population
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                
                # Reproduce (Crossover + Mutation)
                child = GeneticOperators.reproduce(p1, p2, self.mutation_rate)
                
                # 3. Memetic Local Search
                if random.random() < self.local_search_prob:
                    child = MemeticLocalSearch.optimize_genome(child)
                    
                new_population.append(child)
                
            self.population = new_population
            
            # Check if we found a new absolute best
            current_best = min(self.population, key=lambda g: g.fitness)
            if current_best.fitness < best_overall.fitness:
                best_overall = current_best
                
            # 4. Live Progress Tracking (Print every 10 generations)
            if gen % 10 == 0 or gen == self.generations:
                current_max_cost = best_overall.fitness[1]
                current_ratio = current_max_cost / self.target_optimum
                is_valid_str = str(best_overall.is_valid())
                
                print(f"Gen {gen:3d} | Max Cost: {current_max_cost:.2f} | Ratio: {current_ratio:.2f}x | Sum: {best_overall.fitness[2]:.2f} | Valid: {is_valid_str}")

        return best_overall