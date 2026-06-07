import os
import time
import csv
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

# Target optimal values for single TSP instances (used as baselines for ratio calculation)
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
    Executes a single algorithmic run and returns all performance metrics.
    """
    start_time = time.time()
    
    algo = MinimaxMemeticAlgorithm(
        instance=instance, 
        pop_size=pop_size, 
        generations=generations, 
        mutation_rate=0.1, 
        local_search_prob=local_search_prob,
        elitism_count=2,
        target_optimum=target_optimum
    )
    
    # Retrieve all 4 values returned by the updated run() method
    best_solution, conv_gen, final_valid, final_div = algo.run()
    end_time = time.time()
    
    # Extract Lexicographical metrics from the best genome
    max_c, sum_c, diff_c = best_solution.get_lexicographical_scores()
    runtime = end_time - start_time
    
    return max_c, sum_c, diff_c, conv_gen, final_valid, final_div, runtime

def format_float(val: float) -> str:
    """Helper function to ensure all values are consistently formatted with 2 decimal places."""
    return f"{val:.2f}"

def main():
    """
    Main execution script to benchmark the algorithm against the target instances.
    It runs the three required configurations, logs progress, and compiles a final CSV/table.
    """
    instances_to_test = list(BENCHMARKS.keys())
    data_dir = "data"
    csv_filename = "detailed_experiment_results.csv"
    
    pop_size = 50
    generations = 100

    all_results = [] # Stores all results for the final console table

    with open(csv_filename, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        # Headers mapped exactly to the 7 required lab metrics
        writer.writerow(["Instance", "Algorithm", "1. Max Cost", "2. Sum Costs", "3. Imbalance", "4. Conv Gen", "5. Time (s)", "6. Valid Ratio", "7. Diversity", "Ratio"])

        for name in instances_to_test:
            filepath = os.path.join(data_dir, f"{name}.tsp")
            
            if not os.path.exists(filepath):
                print(f"\n[WARNING] File {filepath} not found. Skipping...")
                continue
                
            instance = TSPInstance(filepath)
            target_optimum = BENCHMARKS.get(name, 1.0)
            
            print("\n" + "*"*95)
            print(f" EXPERIMENT: Testing Instance '{name}' (Optimum: {target_optimum})")
            print("*"*95)
            
            configs = [
                ("Standard GA (0%)", 0.0),
                ("Partial Memetic (20%)", 0.2),
                ("Full Memetic (100%)", 1.0)
            ]
            
            for config_name, prob in configs:
                print(f" -> Running: {config_name} (Local Search Prob: {prob}) ...")
                
                # Execute the experiment and gather all statistics
                max_c, sum_c, diff_c, conv_gen, valid_r, div_r, runtime = run_experiment(
                    instance, name, prob, target_optimum, pop_size=pop_size, generations=generations
                )
                
                ratio = max_c / target_optimum
                
                # Format to strings with 2 decimal places to ensure consistency (e.g., 699.00)
                max_c_str = format_float(max_c)
                sum_c_str = format_float(sum_c)
                diff_c_str = format_float(diff_c)
                runtime_str = format_float(runtime)
                ratio_str = format_float(ratio)
                div_r_str = f"{div_r:.4f}"
                
                # Append to memory for the final summary table
                all_results.append((name, config_name, max_c_str, sum_c_str, diff_c_str, conv_gen, runtime_str, valid_r, div_r_str, ratio_str))
                
                # Write row directly to CSV (writing strings ensures Excel/Sheets won't drop the .00)
                writer.writerow([name, config_name, max_c_str, sum_c_str, diff_c_str, conv_gen, runtime_str, valid_r, div_r_str, ratio_str])

    # =========================================================================
    # Print the final detailed table for the lab report
    # =========================================================================
    print("\n\n" + "="*135)
    print(" "*45 + "FINAL DETAILED LAB REPORT TABLE")
    print("="*135)
    
    # Table headers corresponding exactly to the lab report requirements
    header = f"{'Instance':<12} | {'Algorithm Version':<22} | {'1. Max Cost':<11} | {'2. Sum':<8} | {'3. Imbal.':<9} | {'4. Conv.Gen':<11} | {'5. Time(s)':<10} | {'6. Valid %':<10} | {'7. Div.':<7} | {'Ratio':<6}"
    print(header)
    print("-" * 135)
    
    current_instance = ""
    for res in all_results:
        name, config, max_c, sum_c, diff_c, conv_gen, runtime, valid_r, div_r, ratio = res
        
        # Visual separator between different problem instances
        if name != current_instance:
            if current_instance != "":
                print("-" * 135)
            current_instance = name
            display_name = name
        else:
            display_name = "" 

        # Convert the validity ratio to a percentage format (e.g., 1.0 -> 100.0%)
        valid_percent = valid_r * 100
        
        # We use string formatting now because the variables are already perfectly formatted strings
        print(f"{display_name:<12} | {config:<22} | {max_c:<11} | {sum_c:<8} | {diff_c:<9} | {conv_gen:<11d} | {runtime:<10} | {valid_percent:<9.1f}% | {div_r:<7} | {ratio}x")
        
    print("="*135)
    print(f"\n✅ All detailed metrics saved to '{csv_filename}'. Ready for the report!")

if __name__ == "__main__":
    main()