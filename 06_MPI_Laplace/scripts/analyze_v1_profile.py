#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def plot_v1_profile(csv_file):
    """
    Analyze V1 performance profiling data and generate plots
    """
    
    # Read the CSV data
    try:
        df = pd.read_csv(csv_file)
    except FileNotFoundError:
        print(f"Error: Could not find {csv_file}")
        return
    
    # Clean up the matrix_size column (extract just the number)
    df['size'] = df['matrix_size'].str.extract('(\d+)').astype(int)
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('V1 Performance Profile Analysis', fontsize=16)
    
    # Plot 1: Communication overhead scaling
    ax1.set_title('Communication Overhead vs Process Count')
    for size in df['size'].unique():
        size_data = df[df['size'] == size].sort_values('processes')
        ax1.plot(size_data['processes'], size_data['comm_percent'], 
                marker='o', label=f'{size}×{size}', linewidth=2)
    ax1.set_xlabel('Number of Processes')
    ax1.set_ylabel('Communication Time (%)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 50)  # Cap at 50% for readability
    
    # Plot 2: Efficiency degradation
    ax2.set_title('Parallel Efficiency vs Process Count')
    for size in df['size'].unique():
        size_data = df[df['size'] == size].sort_values('processes')
        ax2.plot(size_data['processes'], size_data['efficiency'], 
                marker='s', label=f'{size}×{size}', linewidth=2)
    ax2.set_xlabel('Number of Processes')
    ax2.set_ylabel('Efficiency (Computation/Total)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.0)
    
    # Plot 3: Time distribution stacked bar chart
    ax3.set_title('Time Distribution by Configuration')
    
    # Create stacked bar chart
    configs = []
    comp_times = []
    comm_times = []
    error_times = []
    
    for _, row in df.iterrows():
        configs.append(f"{row['size']}×{row['size']}\n{row['processes']}p")
        comp_times.append(row['comp_percent'])
        comm_times.append(min(row['comm_percent'], 100))  # Cap comm at 100% for visibility
        error_times.append(row['error_percent'])
    
    x = np.arange(len(configs))
    ax3.bar(x, comp_times, label='Computation', color='skyblue')
    ax3.bar(x, comm_times, bottom=comp_times, label='Communication', color='lightcoral')
    ax3.bar(x, error_times, bottom=np.array(comp_times) + np.array(comm_times), 
           label='Error Calculation', color='lightgreen')
    
    ax3.set_xlabel('Configuration')
    ax3.set_ylabel('Time Distribution (%)')
    ax3.set_xticks(x)
    ax3.set_xticklabels(configs, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Scalability comparison
    ax4.set_title('Strong Scaling: Speedup vs Process Count')
    for size in df['size'].unique():
        size_data = df[df['size'] == size].sort_values('processes')
        
        # Calculate speedup relative to 4 processes
        baseline_time = size_data[size_data['processes'] == 4]['total_time'].iloc[0]
        speedup = baseline_time / size_data['total_time']
        
        ax4.plot(size_data['processes'], speedup, 
                marker='D', label=f'{size}×{size}', linewidth=2)
    
    # Add ideal speedup line
    min_proc = df['processes'].min()
    max_proc = df['processes'].max()
    ideal_x = np.linspace(min_proc, max_proc, 100)
    ideal_speedup = ideal_x / min_proc
    ax4.plot(ideal_x, ideal_speedup, '--', color='black', alpha=0.5, label='Ideal')
    
    ax4.set_xlabel('Number of Processes')
    ax4.set_ylabel('Speedup (relative to 4 processes)')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('v1_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.savefig('v1_performance_analysis.pdf', bbox_inches='tight')
    print("Plots saved as v1_performance_analysis.png and v1_performance_analysis.pdf")
    
    # Print key insights
    print("\nKey Performance Insights:")
    print("=" * 50)
    
    # Communication bottleneck
    worst_comm = df.loc[df['comm_percent'].idxmax()]
    print(f"Worst communication overhead: {worst_comm['comm_percent']:.1f}% ")
    print(f"  Configuration: {worst_comm['size']}×{worst_comm['size']} matrix, {worst_comm['processes']} processes")
    
    # Efficiency analysis
    best_eff = df.loc[df['efficiency'].idxmax()]
    worst_eff = df.loc[df['efficiency'].idxmin()]
    print(f"\nBest efficiency: {best_eff['efficiency']:.3f}")
    print(f"  Configuration: {best_eff['size']}×{best_eff['size']} matrix, {best_eff['processes']} processes")
    print(f"Worst efficiency: {worst_eff['efficiency']:.3f}")
    print(f"  Configuration: {worst_eff['size']}×{worst_eff['size']} matrix, {worst_eff['processes']} processes")
    
    # Scaling bottlenecks
    print(f"\nScaling bottlenecks identified:")
    poor_scaling = df[df['processes'] >= 12]
    for _, row in poor_scaling.iterrows():
        if row['comm_percent'] > 30:
            print(f"  High comm overhead: {row['size']}×{row['size']}, {row['processes']}p -> {row['comm_percent']:.1f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 plot_v1_profile.py <csv_file>")
        sys.exit(1)
    
    plot_v1_profile(sys.argv[1])