import os
import time
import math
import threading
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext
import matplotlib.pyplot as plt

# Import our custom classes
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

# Dictionary of optimum targets (Single TSP benchmarks)
BENCHMARKS = {
    "burma14": 3323.0, "gr17": 2085.0, "gr24": 1272.0,
    "fri26": 937.0, "bayg29": 1610.0, "dantzig42": 699.0,
    "att48": 10628.0, "eil51": 426.0, "berlin52": 7542.0, "eil76": 538.0
}

class ThreadSafeConsole(object):
    """ Routes print statements from the algorithm safely into the app's text widget """
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, string):
        self.text_widget.after(0, self._append_text, string)

    def _append_text(self, string):
        self.text_widget.insert(tk.END, string)
        self.text_widget.see(tk.END)

    def flush(self):
        pass

class TSPGuiApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Minimax TSP - Memetic Algorithm Optimizer")
        self.root.geometry("800x650")
        
        self.results = {}  # Stores the best genomes and runtime found
        self.instance = None
        
        self._build_ui()

    def _build_ui(self):
        # 1. Top Frame: Controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        ttk.Label(top_frame, text="Select Instance:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=5)
        
        self.combo_instance = ttk.Combobox(top_frame, values=list(BENCHMARKS.keys()), state="readonly", font=("Arial", 12))
        self.combo_instance.current(0)
        self.combo_instance.pack(side=tk.LEFT, padx=5)
        
        self.btn_solve = ttk.Button(top_frame, text="🚀 Start Optimization", command=self.start_solving)
        self.btn_solve.pack(side=tk.LEFT, padx=20)

        # 2. Middle Frame: Live Console
        mid_frame = ttk.Frame(self.root, padding=10)
        mid_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(mid_frame, text="Generation Tracking (Live Log):", font=("Arial", 12)).pack(anchor="w")
        self.console = scrolledtext.ScrolledText(mid_frame, font=("Courier", 11), bg="black", fg="white")
        self.console.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Route standard output to the text widget
        sys.stdout = ThreadSafeConsole(self.console)

        # 3. Bottom Frame: Visualization Buttons
        self.bottom_frame = ttk.LabelFrame(self.root, text=" View Paths (Visualizations) ", padding=10)
        self.bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.btn_ga = ttk.Button(self.bottom_frame, text="👁️ View GA Path", command=lambda: self.show_paths("GA"))
        self.btn_ga.pack(side=tk.LEFT, padx=5)
        
        self.btn_partial = ttk.Button(self.bottom_frame, text="👁️ View Partial Memetic Path", command=lambda: self.show_paths("Partial"))
        self.btn_partial.pack(side=tk.LEFT, padx=5)
        
        self.btn_full = ttk.Button(self.bottom_frame, text="👁️ View Full Memetic Path", command=lambda: self.show_paths("Full"))
        self.btn_full.pack(side=tk.LEFT, padx=5)
        
        # Initialize state
        self._enable_controls(True)

    def _enable_controls(self, enable_start=True, enable_viz=False):
        """ Robust state management for buttons """
        self.btn_solve.config(state=tk.NORMAL if enable_start else tk.DISABLED)
        state = tk.NORMAL if enable_viz else tk.DISABLED
        self.btn_ga.config(state=state)
        self.btn_partial.config(state=state)
        self.btn_full.config(state=state)

    def start_solving(self):
        # Clear screen and disable solve button, keep viz buttons disabled until done
        self.console.delete(1.0, tk.END)
        self._enable_controls(enable_start=False, enable_viz=False)
        self.results = {}
        
        instance_name = self.combo_instance.get()
        # Run in a separate daemon thread
        threading.Thread(target=self.run_algorithms, args=(instance_name,), daemon=True).start()

    def run_algorithms(self, name):
        filepath = os.path.join("data", f"{name}.tsp")
        if not os.path.exists(filepath):
            print(f"❌ Error: File {filepath} not found!")
            self.root.after(0, lambda: self._enable_controls(True, False))
            return
            
        print(f"Loading instance '{name}'...")
        self.instance = TSPInstance(filepath)
        target = BENCHMARKS.get(name, 1.0)
        
        configs = [
            ("GA", 0.0),
            ("Partial", 0.2),
            ("Full", 1.0)
        ]
        
        try:
            for config_name, prob in configs:
                print(f"\n{'-'*60}")
                print(f"Running algorithm: {config_name} (Local Search Prob: {prob})")
                print(f"{'-'*60}")
                
                algo = MinimaxMemeticAlgorithm(
                    instance=self.instance, pop_size=50, generations=100, 
                    mutation_rate=0.1, local_search_prob=prob, elitism_count=2, target_optimum=target
                )
                
                start_time = time.time()
                best_genome = algo.run()
                end_time = time.time()
                runtime = end_time - start_time
                
                self.results[config_name] = {'genome': best_genome, 'runtime': runtime}
                
                max_c, sum_c, diff_c = best_genome.get_lexicographical_scores()
                print(f"\n✅ Convergence completed for {config_name} in {runtime:.2f} seconds:")
                print(f"   Max Path: {max_c:.2f} | Imbalance: {diff_c:.2f} | Ratio: {max_c/target:.2f}x")
                
            print("\n🎉 All runs completed successfully! You can now view the paths.")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")
        
        # Always re-enable solve button and viz buttons at the end
        self.root.after(0, lambda: self._enable_controls(True, True))

    def show_paths(self, config_name):
        if config_name not in self.results or not self.instance:
            return
            
        data = self.results[config_name]
        genome = data['genome']
        runtime = data['runtime']
        
        coords = self.instance.coords if len(self.instance.coords) == self.instance.num_cities else \
                 [(100*math.cos(2*math.pi*i/self.instance.num_cities), 100*math.sin(2*math.pi*i/self.instance.num_cities)) 
                  for i in range(self.instance.num_cities)]

        path1 = genome.path1 + [genome.path1[0]]
        path2 = genome.path2 + [genome.path2[0]]

        plt.figure(figsize=(10, 6))
        plt.title(f"Minimax TSP Solution - {config_name} ({self.combo_instance.get()})\nExecution Time: {runtime:.2f} seconds", fontweight="bold")
        plt.plot([coords[n][0] for n in path1], [coords[n][1] for n in path1], color='blue', linewidth=2.5, label='Path 1 (Blue)', alpha=0.7)
        plt.plot([coords[n][0] for n in path2], [coords[n][1] for n in path2], color='red', linewidth=2.5, linestyle='--', label='Path 2 (Red)', alpha=0.7)
        plt.scatter([c[0] for c in coords], [c[1] for c in coords], color='black', zorder=5)
        
        for i, (cx, cy) in enumerate(coords):
            plt.text(cx, cy, f" {i}", fontsize=9, zorder=6)

        plt.legend()
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    root = tk.Tk()
    app = TSPGuiApp(root)
    root.mainloop()