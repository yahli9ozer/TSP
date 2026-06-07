import os
import time
import math
import threading
import sys
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt

# Import our custom classes
from src.instance import TSPInstance
from src.main import MinimaxMemeticAlgorithm

# Dictionary of optimum targets (used ONLY for known instances, otherwise ignored)
BENCHMARKS = {
    "burma14": 3323.0, "gr17": 2085.0, "gr24": 1272.0,
    "fri26": 937.0, "bayg29": 1610.0, "dantzig42": 699.0,
    "att48": 10628.0, "eil51": 426.0, "berlin52": 7542.0, "eil76": 538.0
}

def get_available_instances():
    """ Robustly scans the 'data' directory for all .tsp files. """
    instances = []
    
    # Check both relative path and absolute script path
    possible_paths = [
        "data",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
    ]
    
    valid_data_dir = None
    for p in possible_paths:
        if os.path.exists(p) and os.path.isdir(p):
            valid_data_dir = p
            break
            
    if valid_data_dir:
        for f in os.listdir(valid_data_dir):
            if f.lower().endswith(".tsp"):
                instances.append(f[:-4])  # Remove the .tsp extension
                
    if instances:
        return sorted(instances)
        
    print("⚠️ Warning: No .tsp files found in the 'data' folder.")
    return list(BENCHMARKS.keys())  # Fallback only if folder is completely empty

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
        self.root.title("Minimax TSP - Custom Algorithm Optimizer")
        self.root.geometry("850x650")
        
        self.results = {}  # Stores the best genomes and runtime found
        self.instance = None
        
        self._build_ui()

    def _build_ui(self):
        # 1. Top Frame: Controls
        top_frame = ttk.Frame(self.root, padding=10)
        top_frame.pack(fill=tk.X)

        # Instance Selection (Dynamically loaded from data folder)
        ttk.Label(top_frame, text="Instance:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(0, 5))
        
        available_files = get_available_instances()
        self.combo_instance = ttk.Combobox(top_frame, values=available_files, state="readonly", width=15, font=("Arial", 11))
        if available_files:
            self.combo_instance.current(0)
        self.combo_instance.pack(side=tk.LEFT, padx=5)
        
        # Generations Selection
        ttk.Label(top_frame, text="Generations:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.spin_gens = ttk.Spinbox(top_frame, from_=10, to=100000, width=6, font=("Arial", 11))
        self.spin_gens.set(100)  # Default value
        self.spin_gens.pack(side=tk.LEFT, padx=5)

        # Local Search Prob Selection
        ttk.Label(top_frame, text="LS Prob (0-1):", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(15, 5))
        self.spin_prob = ttk.Spinbox(top_frame, from_=0.0, to=1.0, increment=0.1, width=5, font=("Arial", 11))
        self.spin_prob.set(0.2)  # Default value
        self.spin_prob.pack(side=tk.LEFT, padx=5)

        self.btn_solve = ttk.Button(top_frame, text="🚀 Start Run", command=self.start_solving)
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
        
        # Only one button needed now, as we run one custom config at a time
        self.btn_view = ttk.Button(self.bottom_frame, text="👁️ View Path", command=lambda: self.show_paths("Custom"))
        self.btn_view.pack(side=tk.LEFT, padx=5)
        
        # Initialize state
        self._enable_controls(True)

    def _enable_controls(self, enable_start=True, enable_viz=False):
        """ Robust state management for buttons """
        self.btn_solve.config(state=tk.NORMAL if enable_start else tk.DISABLED)
        self.btn_view.config(state=tk.NORMAL if enable_viz else tk.DISABLED)

    def start_solving(self):
        # Validate inputs
        try:
            generations = int(self.spin_gens.get())
            ls_prob = float(self.spin_prob.get())
            if ls_prob < 0.0 or ls_prob > 1.0:
                raise ValueError("Probability must be between 0.0 and 1.0")
        except ValueError as e:
            messagebox.showerror("Invalid Input", "Please enter valid numbers for Generations and LS Prob.")
            return

        # Clear screen and disable solve button, keep viz buttons disabled until done
        self.console.delete(1.0, tk.END)
        self._enable_controls(enable_start=False, enable_viz=False)
        self.results = {}
        
        instance_name = self.combo_instance.get()
        
        # Run in a separate daemon thread
        threading.Thread(target=self.run_algorithms, args=(instance_name, generations, ls_prob), daemon=True).start()

    def run_algorithms(self, name, generations, ls_prob):
        # Determine absolute path to the file
        filepath = os.path.join("data", f"{name}.tsp")
        if not os.path.exists(filepath):
            filepath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", f"{name}.tsp")
            
        if not os.path.exists(filepath):
            print(f"❌ Error: File {filepath} not found!")
            self.root.after(0, lambda: self._enable_controls(True, False))
            return
            
        print(f"Loading instance '{name}'...")
        self.instance = TSPInstance(filepath)
        
        # Check if we have a known target for this instance to calculate ratio
        target = BENCHMARKS.get(name, None)
        
        try:
            print(f"\n{'-'*65}")
            print(f"Running Custom Algorithm: {generations} Gens | Local Search Prob: {ls_prob}")
            print(f"{'-'*65}")
            
            # Pass target=1.0 internally if None to avoid breaking the core algorithm math
            algo_target = target if target else 1.0
            
            algo = MinimaxMemeticAlgorithm(
                instance=self.instance, pop_size=50, generations=generations, 
                mutation_rate=0.1, local_search_prob=ls_prob, elitism_count=2, target_optimum=algo_target
            )
            
            start_time = time.time()
            
            # Safe extraction of the output
            output = algo.run()
            if isinstance(output, tuple):
                best_genome = output[0]  
            else:
                best_genome = output
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # Store the result
            self.results["Custom"] = {
                'genome': best_genome, 
                'runtime': runtime, 
                'gens': generations, 
                'prob': ls_prob
            }
            
            max_c, sum_c, diff_c = best_genome.get_lexicographical_scores()
            print(f"\n✅ Optimization completed in {runtime:.2f} seconds:")
            
            if target:
                print(f"   Max Path: {max_c:.2f} | Imbalance: {diff_c:.2f} | Ratio: {max_c/target:.2f}x")
            else:
                print(f"   Max Path: {max_c:.2f} | Imbalance: {diff_c:.2f}")
                print(f"   (Ratio not calculated: '{name}' is a custom file with no predefined optimum in BENCHMARKS)")
                
            print("\n🎉 Run completed successfully! You can now view the paths.")
            
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
        gens = data['gens']
        prob = data['prob']
        
        coords = self.instance.coords if len(self.instance.coords) == self.instance.num_cities else \
                 [(100*math.cos(2*math.pi*i/self.instance.num_cities), 100*math.sin(2*math.pi*i/self.instance.num_cities)) 
                  for i in range(self.instance.num_cities)]

        path1 = genome.path1 + [genome.path1[0]]
        path2 = genome.path2 + [genome.path2[0]]

        plt.figure(figsize=(10, 6))
        plt.title(f"Minimax TSP Solution - {self.combo_instance.get()}\nGenerations: {gens} | LS Prob: {prob} | Time: {runtime:.2f}s", fontweight="bold")
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