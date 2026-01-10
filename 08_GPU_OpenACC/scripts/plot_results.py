import matplotlib.pyplot as plt
import csv
import sys
import os

# Usage: python plot_results.py <path_to_csv_data>

def plot_performance(data_file):
    labels = []
    times = []
    
    # 1. Read Data
    print(f"Reading data from: {data_file}")
    try:
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            header = next(reader, None) # Skip header safely
            
            for row_num, row in enumerate(reader, start=2):
                if not row: continue # Skip empty lines
                
                try:
                    # Robust parsing
                    label = row[0]
                    time_val = float(row[1])
                    labels.append(label)
                    times.append(time_val)
                except ValueError as e:
                    print(f"⚠️ Warning: Skipping malformed line {row_num}: {row} -> {e}")
                    continue
                    
    except Exception as e:
        print(f"❌ Critical Error reading file: {e}")
        return

    if not times:
        print("❌ Error: No valid data found to plot.")
        return

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Dynamic Coloring logic
    colors = []
    for i, label in enumerate(labels):
        if i == 0:
            colors.append('#A9A9A9') # Gray (Baseline)
        elif i == len(labels) - 1:
            colors.append('#2ca02c') # Green (Final Best)
        elif "Fail" in label or "V3" in label:
            colors.append('#d62728') # Red (Regression)
        else:
            colors.append('#1f77b4') # Blue (Intermediate)

    bars = ax.bar(labels, times, color=colors, width=0.6, edgecolor='black', linewidth=1)
    
    # 3. Formatting
    ax.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax.set_title('Laplace Solver Optimization Journey\n(4096 x 4096 Mesh, 10k Iterations)', fontsize=14, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Set Y-limit
    ax.set_ylim(0, max(times) * 1.15)

    # 4. Add Value Labels & Speedup
    baseline_time = times[0]
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        speedup = baseline_time / height
        
        # Time Label
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{height:.2f} s',
                ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Speedup Label (inside bar)
        # Only show speedup if it's significant and not the baseline
        if i > 0 and abs(speedup - 1.0) > 0.05:
            ax.text(bar.get_x() + bar.get_width()/2., height/2,
                    f'{speedup:.2f}x',
                    ha='center', va='center', fontsize=10, color='white', fontweight='bold')

    # 5. Save Output
    output_dir = '../results/plots'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'performance_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved successfully to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_performance(sys.argv[1])
    else:
        print("Error: Please provide a CSV file path argument.")