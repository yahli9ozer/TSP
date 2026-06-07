import random
from typing import List, Tuple
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
        """Initializes the population and ensures all starting individuals are strictly valid."""
        self.population = []
        for _ in range(self.pop_size):
            genome = TwinPathsGenome.create_random(self.instance)
            if not genome.is_valid():
                genome = GeneticOperators.repair_genome(genome)
            self.population.append(genome)

    def tournament_selection(self, k: int = 3) -> TwinPathsGenome:
        """Selects a parent using tournament selection based on lexicographical fitness."""
        contenders = random.sample(self.population, k)
        return min(contenders, key=lambda g: g.fitness)

    def run(self) -> Tuple[TwinPathsGenome, int, float, float]:
        """
        Executes the evolutionary process and tracks performance metrics.
        Returns:
            Tuple containing: (best_genome, convergence_generation, final_valid_ratio, final_diversity)
        """
        self.initialize_population()
        
        # Initialize best_overall with the best from the starting population
        best_overall = min(self.population, key=lambda g: g.fitness)
        
        print(f"Starting Evolution... Local Search Prob: {self.local_search_prob}")
        
        # Initial logging for Gen 0
        initial_max_cost, _, _ = best_overall.get_lexicographical_scores()
        print(f"Gen   0 | Max Cost: {initial_max_cost:.2f} | Ratio: {initial_max_cost/self.target_optimum:.2f}x")

        convergence_gen = 0
        valid_ratio = 1.0
        diversity = 1.0

        for gen in range(1, self.generations + 1):
            new_population = []
            
            # 1. Elitism
            self.population.sort(key=lambda g: g.fitness)
            new_population.extend(self.population[:self.elitism_count])
            
            # 2. Breeding
            while len(new_population) < self.pop_size:
                p1 = self.tournament_selection()
                p2 = self.tournament_selection()
                
                child = GeneticOperators.reproduce(p1, p2, self.mutation_rate)
                
                # 3. Memetic Local Search
                if random.random() < self.local_search_prob:
                    child = MemeticLocalSearch.optimize_genome(child)
                    
                new_population.append(child)
            
            self.population = new_population
            
            # --- Tracking Metrics ---
            valid_count = sum(1 for g in self.population if g.is_valid())
            valid_ratio = valid_count / self.pop_size
            
            all_edges = set()
            for genome in self.population:
                all_edges.update(genome.edges1)
                all_edges.update(genome.edges2)
            diversity = len(all_edges) / (2 * self.instance.num_cities * self.pop_size)
            
            # Update best_overall and track convergence generation
            current_gen_best = min(self.population, key=lambda g: g.fitness)
            if current_gen_best.fitness < best_overall.fitness:
                best_overall = current_gen_best
                convergence_gen = gen  # Record the generation where the best solution was found
                
            if gen % 10 == 0 or gen == self.generations:
                max_c, sum_c, diff_c = best_overall.get_lexicographical_scores()
                ratio = max_c / self.target_optimum
                
                print(f"Gen {gen:3d} | Max Cost: {max_c:.2f} | Ratio: {ratio:.2f}x | "
                      f"Valid: {valid_ratio:.2f} | Div: {diversity:.3f}")

        # Return the best genome along with the required statistical metrics
        return best_overall, convergence_gen, valid_ratio, diversity