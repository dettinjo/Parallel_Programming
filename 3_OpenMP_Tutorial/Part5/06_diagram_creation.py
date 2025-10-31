import matplotlib.pyplot as plt
import numpy as np

# --- Your Data from Part 5 ---
threads = [1, 2, 4, 6, 8, 12]
wall_time_s = [55.9023, 33.4738, 20.7071, 19.1626, 20.8659, 22.2563]
speedup = [1.00, 1.67, 2.70, 2.92, 2.68, 2.51]
efficiency_pct = [100.0, 83.5, 67.5, 48.6, 33.5, 20.9]
# --- End of Data ---

# Set a clean style for the plots
plt.style.use('ggplot')

# ===================================================================
# Plot 1: Generate the exact 'p5_scaling_graph.png' for your report
# ===================================================================
plt.figure(figsize=(10, 6))
ax1 = plt.gca() # Get current axis

# Plot the measured speedup
ax1.plot(threads, speedup, 'bo-', label='Measured Speedup', markersize=8)

# Plot the ideal linear speedup line (y=x)
ideal_speedup = [t for t in threads]
ax1.plot(threads, ideal_speedup, 'r--', label='Ideal Linear Speedup')

# Add a vertical line at 6 threads (peak performance)
ax1.axvline(x=6, color='gray', linestyle=':', label='Peak Performance (6 Threads)')

# Set titles and labels
ax1.set_title('P5: Speedup vs. Number of Threads', fontsize=16)
ax1.set_xlabel('Number of Threads', fontsize=12)
ax1.set_ylabel('Speedup (vs. 1-thread)', fontsize=12)

# Set x-ticks to match your exact thread counts
ax1.set_xticks(threads)
ax1.set_xticklabels(threads)

# Add grid and legend
ax1.grid(True)
ax1.legend(fontsize=12)

# Save the figure
plt.savefig('p5_scaling_graph.png', dpi=300, bbox_inches='tight')
print(f"Successfully saved 'p5_scaling_graph.png'")


# ===================================================================
# Plot 2: (Combined analysis plot removed as requested)
# ===================================================================
#
# fig, (ax_speedup, ax_efficiency, ax_time) = plt.subplots(3, 1, figsize=(10, 18), sharex=True)
#
# # --- Subplot 1: Speedup ---
# ax_speedup.plot(threads, speedup, 'bo-', label='Measured Speedup', markersize=8)
# ax_speedup.plot(threads, ideal_speedup, 'r--', label='Ideal Linear Speedup')
# ax_speedup.axvline(x=6, color='gray', linestyle=':', label='Peak Performance (6 Threads)')
# ax_speedup.set_title('P5 Performance: Speedup', fontsize=16)
# ax_speedup.set_ylabel('Speedup (vs. 1-thread)', fontsize=12)
# ax_speedup.grid(True)
# ax_speedup.legend(fontsize=12)
#
# # --- Subplot 2: Parallel Efficiency ---
# ax_efficiency.plot(threads, efficiency_pct, 'go-', label='Parallel Efficiency', markersize=8)
# ax_efficiency.axvline(x=6, color='gray', linestyle=':')
# ax_efficiency.set_title('P5 Performance: Efficiency', fontsize=16)
# ax_efficiency.set_ylabel('Efficiency (%)', fontsize=12)
# ax_efficiency.grid(True)
# ax_efficiency.legend(fontsize=12)
# ax_efficiency.set_ylim(0, 110) # Set Y-axis from 0% to 110%
#
# # --- Subplot 3: Absolute Execution Time ---
# ax_time.plot(threads, wall_time_s, 'mo-', label='Execution Time', markersize=8)
# ax_time.axvline(x=6, color='gray', linestyle=':')
# ax_time.set_title('P5 Performance: Execution Time', fontsize=16)
# ax_time.set_ylabel('Wall Time (seconds)', fontsize=12)
# ax_time.set_xlabel('Number of Threads', fontsize=14)
# ax_time.grid(True)
# ax_time.legend(fontsize=12)
#
# # Set x-ticks for all subplots to match your exact thread counts
# plt.xticks(threads)
#
# # Adjust layout and save
# plt.tight_layout()
# plt.savefig('p5_all_plots.png', dpi=300, bbox_inches='tight')
# print(f"Successfully saved 'p5_all_plots.png'")

