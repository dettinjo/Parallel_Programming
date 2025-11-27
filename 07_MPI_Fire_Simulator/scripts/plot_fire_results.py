import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
import glob

# --- Configuration ---
# List of specific job IDs or file patterns to compare for the "Combined Strategy" plot.
# These correspond to your Basic 1D, Optimized 1D, and 2D results.
COMPARISON_CONFIG = [
    {'pattern': '*70908_1d.csv',      'label': 'Basic 1D',          'marker': 'o', 'style': ':',  'color': 'gray'},
    {'pattern': '*70911_1d_opt.csv',  'label': 'Optimized 1D',      'marker': 's', 'style': '-',  'color': 'blue'},
    {'pattern': '*70921_2d.csv',      'label': '2D Decomposition',  'marker': '^', 'style': '--', 'color': 'red'}
]

def get_results_dir():
    """Finds the results directory relative to this script."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))
    return os.path.join(project_root, 'results')

def plot_latest_run(results_dir):
    """Finds the newest CSV and plots all test cases for that specific run."""
    print("\n--- Generating Single Run Plot ---")
    
    # Find all fire timing CSVs
    csv_pattern = os.path.join(results_dir, 'fire_timings_*.csv')
    csv_files = glob.glob(csv_pattern)
    
    if not csv_files:
        print(f"No CSV files found matching {csv_pattern}")
        return

    # Identify the newest file
    latest_csv_path = max(csv_files, key=os.path.getctime)
    print(f"Latest file found: {os.path.basename(latest_csv_path)}")

    try:
        df = pd.read_csv(latest_csv_path)
        # Clean data
        df['speedup'] = pd.to_numeric(df['speedup'], errors='coerce')
        df['processes'] = pd.to_numeric(df['processes'], errors='coerce')
        df = df.dropna(subset=['speedup', 'processes'])

        if df.empty:
            print("File is empty or invalid.")
            return

        # Setup Plot
        plt.figure(figsize=(12, 8))
        plt.style.use('ggplot')
        
        test_cases = sorted(df['test_case'].unique())
        max_procs = df['processes'].max()

        # Plot Ideal Line
        plt.plot([1, max_procs], [1, max_procs], 'k--', alpha=0.5, label='Ideal Linear')

        # Plot each test case
        for case in test_cases:
            df_case = df[df['test_case'] == case]
            mpi_data = df_case[df_case['type'] == 'mpi'].sort_values(by='processes')
            
            if not mpi_data.empty:
                plt.plot(mpi_data['processes'], mpi_data['speedup'], 
                         label=f'Test Case: {case}', marker='o')

        # Formatting
        base_filename = os.path.splitext(os.path.basename(latest_csv_path))[0]
        # Extract job ID for title (assuming format ..._12345...)
        parts = base_filename.split('_')
        job_id = next((s for s in parts if s.isdigit()), 'Unknown')
        
        plt.title(f'Speedup Analysis: Job {job_id}', fontsize=16)
        plt.xlabel('Number of Processes')
        plt.ylabel('Speedup')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Save
        plot_filename = base_filename.replace('timings', 'speedup') + '.png'
        plot_path = os.path.join(results_dir, plot_filename)
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved single run plot: {plot_filename}")
        plt.close()

    except Exception as e:
        print(f"Error processing latest run: {e}")

def plot_strategy_comparison(results_dir):
    """Locates specific legacy files to create the final comparison chart for the report."""
    print("\n--- Generating Strategy Comparison Plot ---")
    
    plt.figure(figsize=(12, 8))
    plt.style.use('ggplot')
    
    max_procs_global = 0
    files_processed = 0

    for config in COMPARISON_CONFIG:
        # Find file by pattern
        pattern_path = os.path.join(results_dir, config['pattern'])
        matches = glob.glob(pattern_path)
        
        if not matches:
            print(f"Warning: Comparison file for '{config['label']}' not found.")
            continue
            
        # Use the first match found
        file_path = matches[0]
        
        try:
            df = pd.read_csv(file_path)
            df['speedup'] = pd.to_numeric(df['speedup'], errors='coerce')
            df = df.dropna(subset=['speedup'])
            
            # Filter for Test 2 only (The main large dataset)
            df_plot = df[(df['test_case'] == 'test2') & (df['type'] == 'mpi')].sort_values(by='processes')
            
            if not df_plot.empty:
                plt.plot(df_plot['processes'], df_plot['speedup'], 
                         label=config['label'], 
                         marker=config['marker'], 
                         linestyle=config['style'],
                         color=config['color'],
                         linewidth=2)
                
                max_procs = df_plot['processes'].max()
                if max_procs > max_procs_global:
                    max_procs_global = max_procs
                
                files_processed += 1
                print(f"Added {config['label']} from {os.path.basename(file_path)}")
                
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    if files_processed > 0:
        # Ideal Line
        plt.plot([0, max_procs_global], [0, max_procs_global], 'k-', alpha=0.1, label='Ideal Linear')
        
        plt.title('Comparison of Parallel Strategies (Test 2 - 8000x8000)', fontsize=16, fontweight='bold')
        plt.xlabel('Number of Processes', fontsize=12)
        plt.ylabel('Speedup Factor', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save
        output_path = os.path.join(results_dir, 'fire_strategy_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved comparison plot: fire_strategy_comparison.png")
    else:
        print("Could not generate comparison plot (no valid data found).")
    
    plt.close()

if __name__ == "__main__":
    res_dir = get_results_dir()
    
    # 1. Create the standard plot for the latest run
    plot_latest_run(res_dir)
    
    # 2. Create the combined comparison plot
    plot_strategy_comparison(res_dir)