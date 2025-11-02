#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import sys

# --- INCREASE FONT SIZES FOR REPORT ---
# Set global font sizes for better readability
plt.rc('font', size=14)          # default text size
plt.rc('axes', titlesize=18)     # title
plt.rc('axes', labelsize=14)     # x and y labels
plt.rc('xtick', labelsize=12)    # x tick labels
plt.rc('ytick', labelsize=12)    # y tick labels
plt.rc('legend', fontsize=12)    # legend
plt.rc('figure', titlesize=18)   # figure title
# --------------------------------------

# Define file names
SPEEDUP_FILE = "07_speedup_data.csv"
BOTTLENECK_FILE = "08_bottleneck_data.csv"

# --- PARSE DATA FROM FILES ---
try:
    speedup_df = pd.read_csv(SPEEDUP_FILE)
    bottleneck_df = pd.read_csv(BOTTLENECK_FILE)
except FileNotFoundError as e:
    print(f"Error: Could not find data file. {e}", file=sys.stderr)
    print("Plot generation failed. Did the C program run successfully?", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error reading CSV data: {e}", file=sys.stderr)
    print("Plot generation failed.", file=sys.stderr)
    sys.exit(1)


# --- PLOT 1: SPEEDUP ANALYSIS ---
print("Generating 09_speedup_analysis.png...")

try:
    # Increased figure size slightly for better spacing with larger fonts
    plt.figure(figsize=(11, 7))

    # Filter out the 'test1' data as it's an overhead case
    plot_data = speedup_df[
        (speedup_df['Version'] == 'OMP_Improved') & 
        (speedup_df['TestFile'] != 'test_files/test1')
    ]

    # Get thread counts
    threads = sorted(plot_data['NumThreads'].unique())

    # Plot a line for each test file
    for test_file in sorted(plot_data['TestFile'].unique()):
        file_data = plot_data[plot_data['TestFile'] == test_file]
        # Clean up label
        label = test_file.replace('test_files/', 'Test Case ')
        # Increase line width and marker size for visibility
        
        # --- REVERTED: Plotting Speedup_x ---
        plt.plot(file_data['NumThreads'], file_data['Speedup_x'], marker='o', label=label, markersize=8, linewidth=2.5)

    # --- MODIFICATION: Removed perfect scaling line ---
    # if threads: 
    #     plt.plot(threads, threads, 'k--', label='Perfect Scaling (y=x)', linewidth=2.5)

    # --- REVERTED: Updated Title and Y-Label ---
    plt.title('Parallel Speedup vs. Number of Threads (Relative to Sequential Baseline)')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (x)')
    
    # --- MODIFICATION: Set custom Y-axis scale to show differences better ---
    # Based on your data (min 1.78x, max 4.38x), this zooms in.
    plt.ylim(1.5, 4.5) 
    
    if threads: 
        plt.xticks(threads)
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.tight_layout() # Adjust layout
    
    # --- REVERTED: Changed save file name ---
    plt.savefig('09_speedup_analysis.png')
    plt.close()
    
    print("... saved 09_speedup_analysis.png")

except Exception as e:
    print(f"Error generating speedup plot: {e}", file=sys.stderr)


# --- PLOT 2: BOTTLENECK ANALYSIS ---
# (This plot remains unchanged)
print("Generating 10_bottleneck_analysis.png...")

try:
    plt.figure(figsize=(12, 8)) # Increased figure size

    # Filter to get just the interesting comparisons
    seq_data = bottleneck_df[
        (bottleneck_df['Version'] == 'Sequential') &
        (bottleneck_df['TestFile'].isin(['test_files/test2', 'test_files/test4']))
    ]

    # Get the OMP data for the *highest thread count* tested
    max_threads = bottleneck_df[bottleneck_df['Version'] == 'OMP_Improved']['NumThreads'].max()
    
    omp_data = bottleneck_df[
        (bottleneck_df['Version'] == 'OMP_Improved') &
        (bottleneck_df['NumThreads'] == max_threads) &
        (bottleneck_df['TestFile'].isin(['test_files/test2', 'test_files/test4']))
    ]

    # Combine and sort them
    plot_data = pd.concat([
        seq_data[seq_data['TestFile'] == 'test_files/test2'],
        omp_data[omp_data['TestFile'] == 'test_files/test2'],
        seq_data[seq_data['TestFile'] == 'test_files/test4'],
        omp_data[omp_data['TestFile'] == 'test_files/test4']
    ])

    if not plot_data.empty:
        # Create labels for the x-axis
        labels = [
            'Test 2\n(Sequential)',
            f'Test 2\n(OMP @ {max_threads} Threads)',
            'Test 4\n(Sequential)',
            f'Test 4\n(OMP @ {max_threads} Threads)'
        ]

        # Get data for stacking
        focal_times = plot_data['Time_Focal_s']
        heat_times = plot_data['Time_Heat_s']
        move_times = plot_data['Time_Move_s']
        action_times = plot_data['Time_Action_s']

        # Create the stacked bar chart
        width = 0.6
        p1 = plt.bar(labels, focal_times, width, label='Focal')
        p2 = plt.bar(labels, heat_times, width, bottom=focal_times, label='Heat')
        p3 = plt.bar(labels, move_times, width, bottom=focal_times + heat_times, label='Move')
        p4 = plt.bar(labels, action_times, width, bottom=focal_times + heat_times + move_times, label='Action')

        plt.title(f'Bottleneck Analysis: Time per Section (Sequential vs. OMP @ {max_threads} Threads)')
        plt.ylabel('Total Time (s)')
        plt.legend(loc='upper left')

        # Add total time labels on top of the bars
        totals = focal_times + heat_times + move_times + action_times
        for i, total in enumerate(totals):
            # --- Added fontsize=12 ---
            plt.text(i, total + 1, f'{total:.1f}s', ha='center', fontweight='bold', fontsize=12)

        plt.tight_layout()
        plt.savefig('10_bottleneck_analysis.png')
        plt.close()
        
        print("... saved 10_bottleneck_analysis.png")
    else:
        print("... skipping 10_bottleneck_analysis.png (no data to plot).")

except Exception as e:
    print(f"Error generating bottleneck plot: {e}", file=sys.stderr)


print("\nPlot generation complete.")