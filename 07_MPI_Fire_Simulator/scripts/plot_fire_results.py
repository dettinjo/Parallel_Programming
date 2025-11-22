import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

# --- 1. Define File Paths ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(project_root, 'results')

    # Find the latest FIRE CSV file by pattern
    csv_pattern = os.path.join(results_dir, 'fire_timings_*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern '{csv_pattern}'")
        sys.exit(1)

    # Find the newest file
    latest_csv_path = max(csv_files, key=os.path.getctime)
    print(f"Loading latest data from: {latest_csv_path}")

    # Create a plot name based on the CSV's job ID
    base_filename = os.path.splitext(os.path.basename(latest_csv_path))[0]
    # e.g., "fire_timings_70050" -> "fire_speedup_70050.png"
    plot_filename = base_filename.replace('timings', 'speedup') + '.png'
    plot_path = os.path.join(results_dir, plot_filename)

except Exception as e:
    print(f"Error setting up paths: {e}")
    sys.exit(1)

# --- 2. Load and Clean Data ---
try:
    df = pd.read_csv(latest_csv_path)
    df['speedup'] = pd.to_numeric(df['speedup'], errors='coerce')
    df = df.dropna(subset=['speedup'])
    
    if df.empty:
        print("Error: The CSV file is empty or contains no valid speedup data.")
        sys.exit(1)

except Exception as e:
    print(f"Error reading or cleaning CSV file: {e}")
    sys.exit(1)

# --- 3. Prepare for Plotting ---
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 8))
# Group by 'test_case'
test_cases = sorted(df['test_case'].unique())
max_processes = df['processes'].max()

# --- 4. Plot Ideal Speedup Line ---
ax.plot([1, max_processes], [1, max_processes], 'k--', label='Ideal Speedup')

# --- 5. Loop and Plot Data for Each Test Case ---
for case in test_cases:
    # Filter by 'test_case'
    df_case = df[df['test_case'] == case]
    mpi_data = df_case[df_case['type'] == 'mpi'].sort_values(by='processes')
    seq_data = df_case[df_case['type'] == 'sequential']

    if not mpi_data.empty:
        # Update label
        ax.plot(mpi_data['processes'], mpi_data['speedup'], 
                label=f'Test Case: {case}', marker='o', linestyle='-')
    
    if not seq_data.empty:
        ax.plot(seq_data['processes'], seq_data['speedup'], marker='o', markersize=8)

# --- 6. Format the Plot ---
# Get the Job ID from the filename for the title
job_id = base_filename.split('_')[-1]
# Update title
ax.set_title(f'MPI Fire Simulator: Strong Scaling Speedup (Job {job_id})', fontsize=16, fontweight='bold')

ax.set_xlabel('Number of Processes', fontsize=12)
ax.set_ylabel('Speedup (T_sequential / T_parallel)', fontsize=12)

all_procs = sorted(df[df['type']=='mpi']['processes'].unique())
ax.set_xticks([1] + all_procs)
ax.set_xticklabels([1] + all_procs, rotation=45) 

ax.legend(title='Experiment')
ax.grid(True, which='both', linestyle='--', linewidth=0.5)
ax.set_xlim(left=0, right=max_processes + 1)
ax.set_ylim(bottom=0)

# --- 7. Save and Show ---
try:
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Plot saved to: {plot_path}")
    
except Exception as e:
    print(f"Error saving plot: {e}")