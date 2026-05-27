import time
import tsp_viewer  # Make sure the file is named tsp_viewer.py
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

def main():
    """
    Main execution script for running a single TSP instance and visualizing the results.
    It runs the Minimax Memetic Algorithm and plots the two edge-disjoint paths.
    """
    # Note: Choose the instance you want to test (e.g., "burma14" or "eil76")
    instance_name = "burma14"
    filepath = f"data/{instance_name}.tsp"
    
    try:
        # 1. Load the dataset using the provided TSP viewer logic
        instance = TSPInstance(filepath)
        print(f"Loaded instance '{instance_name}' with {instance.num_cities} cities.")
        
        # 2. Create an instance of the Memetic Algorithm
        # Here we configure a Full Memetic Algorithm by setting local_search_prob to 1.0
        ma = MinimaxMemeticAlgorithm(
            instance=instance,
            pop_size=30,
            generations=50,
            mutation_rate=0.1,
            local_search_prob=1.0,  # 1.0 = Full Memetic Algorithm
            elitism_count=2
        )
        
        # 3. Run the evolution process and measure execution time
        start_time = time.time()
        best_solution = ma.run()
        end_time = time.time()
        
        # 4. Print the final results
        print("\n" + "="*40)
        print("🏆 EVOLUTION COMPLETE 🏆")
        print("="*40)
        
        # Extract the lexicographical evaluation metrics
        max_c, sum_c, diff_c = best_solution.get_lexicographical_scores()
        
        print(f"Best Target (Max Path): {max_c:.2f}")
        print(f"Sum of Paths:           {sum_c:.2f}")
        print(f"Imbalance (|T1 - T2|):  {diff_c:.2f}")
        print(f"Is strictly valid?      {best_solution.is_valid()}")
        print(f"Time elapsed:           {end_time - start_time:.2f} seconds")
        
        # 5. Display the visual graph (both paths in the same window)
        print("\nOpening viewer for both paths...")
        tsp_viewer.plot_twin_tours(
            coords=instance.coords, 
            tour1=best_solution.path1, 
            tour2=best_solution.path2, 
            title=f"Minimax TSP ({instance_name}) | Max Cost: {max_c:.2f} | Imbalance: {diff_c:.2f}"
        )

    except FileNotFoundError:
        print(f"Error: File '{filepath}' not found. Make sure it is in the 'data' folder.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()