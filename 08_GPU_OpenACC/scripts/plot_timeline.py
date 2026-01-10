import sqlite3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys
import numpy as np

# Usage: python plot_timeline.py <baseline.sqlite> <optimized.sqlite>

def get_kernel_data(sqlite_path, skip_kernels=50, window_ms=30):
    """
    Extracts kernel data, skipping the startup phase to find the steady state.
    Returns: list of (start_ms, duration_ms, is_stencil)
    """
    data = []
    try:
        conn = sqlite3.connect(sqlite_path)
        cursor = conn.cursor()
        
        # Get all kernels ordered by time
        # We fetch Global Start/End (ns)
        cursor.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start ASC")
        all_rows = cursor.fetchall()
        conn.close()
        
        if not all_rows:
            return []

        # 1. Skip Initialization (Warmup)
        # If we have enough kernels, skip the first few to hit the loop body
        start_idx = min(len(all_rows) - 1, skip_kernels)
        active_rows = all_rows[start_idx:]
        
        if not active_rows:
            return []

        # Normalize time relative to the first "active" kernel
        # This aligns the two plots visually, regardless of startup time differences
        t0 = active_rows[0][0]
        
        # 2. Heuristic: Identify Stencil vs Other
        # The Stencil kernel is the most common one. We find the median duration.
        durations = [(r[1] - r[0]) for r in active_rows[:200]] # Sample first 200
        median_dur = np.median(durations)
        
        # Tolerance for "same kernel type" (e.g., +/- 20%)
        tolerance = median_dur * 0.2

        limit_ns = window_ms * 1_000_000

        for r in active_rows:
            start_ns = r[0] - t0
            end_ns = r[1] - t0
            duration = r[1] - r[0]
            
            # Stop if we exceed the time window
            if start_ns > limit_ns:
                break
                
            # Classify
            is_stencil = abs(duration - median_dur) < tolerance
            
            # Convert to ms
            data.append({
                'start': start_ns / 1e6,
                'duration': duration / 1e6,
                'is_stencil': is_stencil
            })
            
    except Exception as e:
        print(f"Error reading {sqlite_path}: {e}")
        
    return data

def plot_comparison(base_file, opt_file):
    print("Extracting Smart Timeline Data...")
    # Skip first 100 kernels to ensure we are deep inside the loop
    # Show a 40ms window
    base_data = get_kernel_data(base_file, skip_kernels=100, window_ms=40) 
    opt_data = get_kernel_data(opt_file, skip_kernels=100, window_ms=40)

    if not base_data or not opt_data:
        print("❌ Error: Not enough data to generate timeline.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), sharex=True)
    
    # --- Plot Baseline ---
    ax1.set_title("Baseline (Static): Overhead Gaps", fontsize=11, fontweight='bold', loc='left')
    ax1.set_ylabel("GPU Activity")
    ax1.set_yticks([])
    ax1.set_ylim(0, 1)
    
    for k in base_data:
        # Baseline often has Stencil, Error, Copy. 
        # Color: Gray for Stencil, Red/Orange for others
        color = '#7f7f7f' if k['is_stencil'] else '#ff7f0e'
        rect = patches.Rectangle((k['start'], 0.2), k['duration'], 0.6, facecolor=color, edgecolor='black', linewidth=0.5)
        ax1.add_patch(rect)
        
    # --- Plot Optimized ---
    ax2.set_title("Optimized (V4): Latency Hiding (Packed Execution)", fontsize=11, fontweight='bold', loc='left')
    ax2.set_xlabel("Time (milliseconds) - Normalized to Steady State", fontsize=10)
    ax2.set_ylabel("GPU Activity")
    ax2.set_yticks([])
    ax2.set_ylim(0, 1)
    
    for k in opt_data:
        # Optimized has Stencil (Green) and rare Reduction (Red)
        color = '#2ca02c' if k['is_stencil'] else '#d62728'
        # Make the rare reduction pop out
        h = 0.6 if k['is_stencil'] else 0.8
        y = 0.2 if k['is_stencil'] else 0.1
        
        rect = patches.Rectangle((k['start'], y), k['duration'], h, facecolor=color, edgecolor='black', linewidth=0.5)
        ax2.add_patch(rect)

        # Annotate the reduction if found
        if not k['is_stencil']:
             ax2.annotate('Periodic Check', (k['start'], 0.9), xytext=(k['start'], 1.2),
                          arrowprops=dict(facecolor='black', arrowstyle='->'), ha='center', fontsize=8)

    ax1.grid(axis='x', linestyle='--', alpha=0.3)
    ax2.grid(axis='x', linestyle='--', alpha=0.3)
    
    # Set limit to the max data point we gathered
    max_t = max(base_data[-1]['start'], opt_data[-1]['start'])
    ax2.set_xlim(0, max_t) 
    
    plt.tight_layout()
    
    output_path = '../results/plots/timeline_comparison.png'
    plt.savefig(output_path, dpi=300)
    print(f"✅ Smart Timeline saved to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python plot_timeline.py <base.sqlite> <opt.sqlite>")
    else:
        plot_comparison(sys.argv[1], sys.argv[2])