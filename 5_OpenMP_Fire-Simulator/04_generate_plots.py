#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import io
import numpy as np
import sys

# Define file names
SPEEDUP_FILE = "speedup_data.csv"
BOTTLENECK_FILE = "bottleneck_data.csv"

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
print("Generating speedup_analysis.png...")

try:
    plt.figure(figsize=(10, 6))

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
        plt.plot(file_data['NumThreads'], file_data['Speedup_x'], marker='o', label=label)

    # Plot perfect scaling line
    if threads: # Only plot if 'threads' list is not empty
        plt.plot(threads, threads, 'k--', label='Perfect Scaling (y=x)')

    plt.title('Parallel Speedup vs. Number of Threads (Relative to Sequential Baseline)')
    plt.xlabel('Number of Threads')
    plt.ylabel('Speedup (x)')
    if threads: # Only set xticks if 'threads' list is not empty
        plt.xticks(threads)
    plt.legend()
    plt.grid(True, linestyle=':')
    plt.savefig('speedup_analysis.png')
    plt.close()
    
    print("... saved speedup_analysis.png")

except Exception as e:
    print(f"Error generating speedup plot: {e}", file=sys.stderr)


# --- PLOT 2: BOTTLENECK ANALYSIS ---
print("Generating bottleneck_analysis.png...")

try:
    plt.figure(figsize=(12, 7))

    # Filter to get just the interesting comparisons
    seq_data = bottleneck_df[
        (bottleneck_df['Version'] == 'Sequential') &
        (bottleneck_df['TestFile'].isin(['test_files/test2', 'test_files/test4']))
    ]

    omp_data = bottleneck_df[
        (bottleneck_df['Version'] == 'OMP_Improved') &
        (bottleneck_df['NumThreads'] == 12) &
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
            'Test 2\n(OMP @ 12 Threads)',
            'Test 4\n(Sequential)',
            'Test 4\n(OMP @ 12 Threads)'
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

        plt.title('Bottleneck Analysis: Time per Section (Sequential vs. OMP @ 12 Threads)')
        plt.ylabel('Total Time (s)')
        plt.legend(loc='upper left')

        # Add total time labels on top of the bars
        totals = focal_times + heat_times + move_times + action_times
        for i, total in enumerate(totals):
            plt.text(i, total + 1, f'{total:.1f}s', ha='center', fontweight='bold')

        plt.tight_layout()
        plt.savefig('bottleneck_analysis.png')
        plt.close()
        
        print("... saved bottleneck_analysis.png")
    else:
        print("... skipping bottleneck_analysis.png (no data to plot).")

except Exception as e:
    print(f"Error generating bottleneck plot: {e}", file=sys.stderr)


print("\nPlot generation complete.")