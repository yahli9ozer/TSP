import time
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

def run_experiment(instance: TSPInstance, name: str, ls_prob: float, pop_size=50, generations=100):
    """
    Runs a specific configuration of the Memetic Algorithm.
    Returns a dictionary containing the performance metrics and execution time.
    """
    print(f"\n--- Running: {name} (Local Search Prob: {ls_prob}) ---")
    
    ma = MinimaxMemeticAlgorithm(
        instance=instance,
        pop_size=pop_size,
        generations=generations,
        mutation_rate=0.1,
        local_search_prob=ls_prob,
        elitism_count=2
    )
    
    start_time = time.time()
    best_solution = ma.run()
    end_time = time.time()
    
    max_c, sum_c, diff_c = best_solution.get_lexicographical_scores()
    
    return {
        "Name": name,
        "Max Cost": max_c,
        "Sum": sum_c,
        "Imbalance": diff_c,
        "Valid": best_solution.is_valid(),
        "Time (s)": end_time - start_time
    }

def main():
    # Using a small instance to get immediate results for the comparison table
    filename = "burma14"
    filepath = f"data/{filename}.tsp"
    
    try:
        instance = TSPInstance(filepath)
        print(f"Loaded instance '{filename}' with {instance.num_cities} cities.")
        
        # Define the configurations to compare: Standard GA vs Partial MA vs Full MA
        configs = [
            ("Standard GA (0%)", 0.0),
            ("Partial Memetic (20%)", 0.2),
            ("Full Memetic (100%)", 1.0)
        ]
        
        results = []
        
        # Run the 3 configurations - using a small population and enough generations to see convergence
        for name, prob in configs:
            res = run_experiment(instance, name, prob, pop_size=30, generations=50)
            results.append(res)
            
        # Print a neatly formatted summary table for the final report
        print("\n" + "="*85)
        print(f"{'Algorithm Version':<25} | {'Max Cost':<10} | {'Sum Costs':<10} | {'Imbalance':<10} | {'Valid?':<8} | {'Time (s)':<10}")
        print("-" * 85)
        for r in results:
            print(f"{r['Name']:<25} | {r['Max Cost']:<10.2f} | {r['Sum']:<10.2f} | {r['Imbalance']:<10.2f} | {str(r['Valid']):<8} | {r['Time (s)']:<10.2f}")
        print("="*85)
        
    except FileNotFoundError:
        print(f"Error: Could not find '{filepath}'. Please check your data folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()