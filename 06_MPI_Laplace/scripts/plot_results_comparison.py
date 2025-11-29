import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import glob

# --- 1. Define File Paths ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    results_dir = os.path.join(project_root, 'results')

    # --- FIX: Find the latest CSV file by pattern ---
    csv_pattern = os.path.join(results_dir, 'laplace_timings_*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"Error: No CSV files found matching pattern '{csv_pattern}'")
        sys.exit(1)

    # Find the newest file
    v1_csv_path = '/home/master/ppm/ppm-22/Escritorio/Parallel_Programming/06_MPI_Laplace/results/laplace_timings_clus_72055.csv'
    v2_csv_path = '/home/master/ppm/ppm-22/Escritorio/Parallel_Programming/06_MPI_Laplace/results/laplace_timings_clus_v3_73243.csv'
    overide_csv_path = '/home/master/ppm/ppm-22/Escritorio/Parallel_Programming/06_MPI_Laplace/results/laplace_timings_clus_72049.csv'
    latest_csv_path = max(csv_files, key=os.path.getctime)
    print(f"Loading latest data from: {latest_csv_path}")

    # --- FIX: Create a plot name based on the CSV's job ID ---
    base_filename = os.path.splitext(os.path.basename(latest_csv_path))[0]
    # e.g., "laplace_timings_70035" -> "laplace_speedup_70035.png"
    # plot_filename = base_filename.replace('timings', 'speedup') + 'V2_Compparison.png'
    plot_filename = 'V3-v1_Comparison.png'
    plot_path = os.path.join(results_dir, plot_filename)

except Exception as e:
    print(f"Error setting up paths: {e}")
    sys.exit(1)

# --- 2. Load and Clean Data ---
try:
    
    df_v1 = pd.read_csv(v1_csv_path)
    df_v1['speedup'] = pd.to_numeric(df_v1['speedup'], errors='coerce')
    df_v1 = df_v1.dropna(subset=['speedup'])
    
    df = pd.read_csv(v2_csv_path)
    df['speedup'] = pd.to_numeric(df['speedup'], errors='coerce')
    df = df.dropna(subset=['speedup'])
    

    
    if df.empty or df_v1.empty:
        print("Error: The CSV file is empty or contains no valid speedup data.")
        sys.exit(1)

except Exception as e:
    print(f"Error reading or cleaning CSV file: {e}")
    sys.exit(1)

# --- 3. Prepare for Plotting ---
plt.style.use('ggplot')
fig, ax = plt.subplots(figsize=(12, 8))
ax1 = plt.twinx(ax)
matrix_sizes = sorted(df['matrix_size'].unique())
max_processes = df['processes'].max()

# --- 4. Plot Ideal Speedup Line ---
ax.plot([1, max_processes], [1, max_processes], 'k--', label='Ideal Speedup')
ax.set_ylim(0,100)
ax1.set_ylim(0,100)
# --- 5. Loop and Plot Data for Each Matrix Size ---
colors_matrix = plt.cm.plasma(np.linspace(0, 1, len(matrix_sizes)))
for i, size in enumerate(matrix_sizes):
    df_size = df[df['matrix_size'] == size]
    mpi_data = df_size[df_size['type'] == 'mpi'].sort_values(by='processes')
    seq_data = df_size[df_size['type'] == 'sequential']
    
    df_v1_size = df_v1[df_v1['matrix_size'] == size]
    mpi_v1_data = df_v1_size[df_v1_size['type'] == 'mpi'].sort_values(by='processes')
    seq_v1_data = df_v1_size[df_v1_size['type'] == 'sequential']
    
    if not mpi_data.empty:
        ax.plot(mpi_data['processes'], 
                mpi_data['speedup'], 
                label=f'Size: {size}x{size}', 
                marker='o',
                color=colors_matrix[i],
                linestyle='-',
                alpha = 0.9)
        
        ax1.plot(mpi_v1_data['processes'], 
                mpi_v1_data['speedup'], 
                label=f'Size: {size}x{size}', 
                marker='.',
                color=colors_matrix[i],
                linestyle='dotted',
                alpha = 0.9)
    
    # if not seq_data.empty:
    #     ax.plot(seq_data['processes'], seq_data['speedup'], marker='o', markersize=8)

# --- 6. Format the Plot ---
# Get the Job ID from the filename for the title
job_id = base_filename.split('_')[-1]
ax.set_title(f'MPI Laplace: Strong Scaling Speedup (Job {job_id})', fontsize=16, fontweight='bold')

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