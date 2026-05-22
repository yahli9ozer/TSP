import time
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

try:
    instance = TSPInstance("data/burma14.tsp")
    print(f"Loaded instance with {instance.num_cities} cities.")
    
    # 1. Configuration for FULL Memetic Algorithm
    # pop_size=30 and generations=50 is enough for a small instance like burma14
    ma = MinimaxMemeticAlgorithm(
        instance=instance,
        pop_size=30,
        generations=50,
        mutation_rate=0.1,
        local_search_prob=1.0,  # 1.0 = Full Memetic Algorithm (Every child gets 2-opt)
        elitism_count=2
    )
    
    start_time = time.time()
    best_solution = ma.run()
    end_time = time.time()
    
    print("\n" + "="*40)
    print("🏆 EVOLUTION COMPLETE 🏆")
    print("="*40)
    
    max_c, sum_c, diff_c = best_solution.get_lexicographical_scores()
    print(f"Best Target (Max Path): {max_c:.2f}")
    print(f"Sum of Paths:           {sum_c:.2f}")
    print(f"Imbalance (|T1 - T2|):  {diff_c:.2f}")
    print(f"Is strictly valid?      {best_solution.is_valid()}")
    print(f"Time elapsed:           {end_time - start_time:.2f} seconds")

except Exception as e:
    print(f"An error occurred during evolution: {e}")