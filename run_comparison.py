import os
import time
import csv
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

# Shay's Benchmarks: Single TSP optimum for each instance (from the assignment table)
BENCHMARKS = {
    "burma14": 3323.0,
    "gr17": 2085.0,
    "gr24": 1272.0,
    "fri26": 937.0,
    "bayg29": 1610.0,
    "dantzig42": 699.0,
    "att48": 10628.0,
    "eil51": 426.0,
    "berlin52": 7542.0,
    "eil76": 538.0
}

def run_experiment(instance: TSPInstance, name: str, local_search_prob: float, target_optimum: float, pop_size: int = 50, generations: int = 100):
    """
    Executes a single run of the Minimax Memetic Algorithm.
    
    Args:
        instance: The parsed TSP instance.
        name: The name of the instance (e.g., 'gr24').
        local_search_prob: Probability of applying 2-opt local search (0.0 for Standard GA).
        target_optimum: The baseline optimum for the live ratio tracking.
        pop_size: The size of the population.
        generations: Number of generations to evolve.
        
    Returns:
        A tuple containing (max_cost, sum_cost, imbalance, is_valid, runtime_seconds).
    """
    start_time = time.time()
    
    # Initialize the algorithm engine, now passing target_optimum for live tracking
    algo = MinimaxMemeticAlgorithm(
        instance=instance, 
        pop_size=pop_size, 
        generations=generations, 
        mutation_rate=0.1, 
        local_search_prob=local_search_prob,
        elitism_count=2,
        target_optimum=target_optimum
    )
    
    # Run the evolutionary process
    best_solution = algo.run()
    end_time = time.time()
    
    # Extract metrics for reporting
    max_c, sum_c, diff_c = best_solution.get_lexicographical_scores()
    is_valid = best_solution.is_valid()
    runtime = end_time - start_time
    
    return max_c, sum_c, diff_c, is_valid, runtime

def main():
    """
    Main execution script to benchmark the algorithm against Shay's target instances.
    It runs Standard GA, Partial Memetic, and Full Memetic configurations for each file,
    prints the progress, and saves all metrics to a CSV file.
    """
    # The complete list of instances required by the assignment benchmarks
    instances_to_test = list(BENCHMARKS.keys())
    
    data_dir = "data"
    csv_filename = "experiment_results.csv"
    
    # Optimal parameters found during tuning
    pop_size = 50
    generations = 100

    # Open CSV file for logging the final data
    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Write CSV Header
        writer.writerow(["Instance", "Cities", "Algorithm", "Max Cost", "Sum Costs", "Imbalance", "Valid", "Time (s)", "Optimum Target", "Ratio to Opt"])

        for name in instances_to_test:
            filepath = os.path.join(data_dir, f"{name}.tsp")
            
            # Skip if the TSPLIB file is missing from the data folder
            if not os.path.exists(filepath):
                print(f"\n[WARNING] File {filepath} not found. Skipping...")
                continue
                
            print("\n" + "*"*95)
            print(f" EXPERIMENT: Testing Instance '{name}' ")
            print("*"*95)
            
            # Parse the instance (supports FULL_MATRIX, LOWER_DIAG_ROW, Euclidean, etc.)
            instance = TSPInstance(filepath)
            print(f"Loaded instance '{name}' with {instance.num_cities} cities.")
            
            # Retrieve the target optimum for this specific problem
            target_optimum = BENCHMARKS.get(name, 1.0)
            print(f"Target Single TSP Optimum: {target_optimum}\n")
            
            results = []
            
            # Define the configurations to test
            configs = [
                ("Standard GA (0%)", 0.0),
                ("Partial Memetic (20%)", 0.2),
                ("Full Memetic (100%)", 1.0)
            ]
            
            for config_name, prob in configs:
                print(f" -> Running: {config_name} (Local Search Prob: {prob}) ...")
                
                max_c, sum_c, diff_c, is_valid, runtime = run_experiment(
                    instance, name, prob, target_optimum, pop_size=pop_size, generations=generations
                )
                
                # Calculate final ratio compared to Shay's Single TSP Optimum
                ratio = max_c / target_optimum
                
                # Store for console printing
                results.append((config_name, max_c, sum_c, diff_c, is_valid, runtime, ratio))
                
                # Log to CSV
                writer.writerow([name, instance.num_cities, config_name, round(max_c, 2), round(sum_c, 2), round(diff_c, 2), is_valid, round(runtime, 2), target_optimum, round(ratio, 2)])
                
            # Print the final summary table for the current instance
            print("\n" + "="*95)
            print(f"{'Algorithm Version':<25} | {'Max Cost':<10} | {'Sum Costs':<10} | {'Imbalance':<10} | {'Time (s)':<10} | {'Ratio to Opt'}")
            print("-" * 95)
            for res in results:
                config_name, max_c, sum_c, diff_c, is_valid, runtime, ratio = res
                
                # Add visual cue if we beat the Aspirational Target (1.6x)
                target_status = "🏆 SUPERB!" if ratio <= 1.6 else ("⚠️ OK" if ratio <= 2.0 else "❌ FAILED")
                if name == "burma14":
                    target_status = "ℹ️ GEO Metric"
                    
                print(f"{config_name:<25} | {max_c:<10.2f} | {sum_c:<10.2f} | {diff_c:<10.2f} | {runtime:<10.2f} | {ratio:<5.2f}x  {target_status}")
            print("="*95 + "\n")

    print(f"\n✅ All experiments finished! Full tracking log saved to '{csv_filename}'.")

if __name__ == "__main__":
    main()