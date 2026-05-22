import time
import tsp_viewer  # Make sure the file is named tsp_viewer.py
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

def main():
    # שים לב לבחור את הקובץ שתרצה לבדוק (burma14 או eil76)
    instance_name = "burma14"
    filepath = f"data/{instance_name}.tsp"
    
    try:
        # 1. טעינת הנתונים דרך ה-viewer הנתון
        instance = TSPInstance(filepath)
        print(f"Loaded instance '{instance_name}' with {instance.num_cities} cities.")
        
        # 2. יצירת מופע של האלגוריתם הממטי
        # כאן אנחנו מגדירים את ה-Full Memetic (local_search_prob=1.0)
        ma = MinimaxMemeticAlgorithm(
            instance=instance,
            pop_size=30,
            generations=50,
            mutation_rate=0.1,
            local_search_prob=1.0,  # 1.0 = Memetic מלא
            elitism_count=2
        )
        
        # 3. הפעלת האבולוציה עם מדידת זמנים
        start_time = time.time()
        best_solution = ma.run()
        end_time = time.time()
        
        # 4. הדפסת תוצאות
        print("\n" + "="*40)
        print("🏆 EVOLUTION COMPLETE 🏆")
        print("="*40)
        
        # שליפת מדדי ההערכה
        max_c, sum_c, diff_c = best_solution.get_lexicographical_scores()
        
        print(f"Best Target (Max Path): {max_c:.2f}")
        print(f"Sum of Paths:           {sum_c:.2f}")
        print(f"Imbalance (|T1 - T2|):  {diff_c:.2f}")
        print(f"Is strictly valid?      {best_solution.is_valid()}")
        print(f"Time elapsed:           {end_time - start_time:.2f} seconds")
        
        # 5. הצגת הגרף הויזואלי (שני המסלולים באותו חלון)
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