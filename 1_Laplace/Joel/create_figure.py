import matplotlib.pyplot as plt
import numpy as np

# =============================================================================
# 1. DATA PREPARATION
# =============================================================================
grid_sizes = ['100x100', '1000x1000', '4096x4096']
version_labels = ['V1 (Static)', 'V2 (Dynamic)', 'V3 (Optimized)']

raw_times_v1 = ['0m0.085s', '0m18.868s', '4m23.938s']
raw_times_v2 = ['0m0.087s', '0m18.890s', '2m34.167s']
raw_times_v3 = ['0m0.079s', '0m18.327s', '2m18.616s']

# =============================================================================
# 2. HELPER FUNCTION TO CONVERT TIME STRINGS TO SECONDS
# =============================================================================
def time_to_seconds(time_str):
    """Converts a 'MmSS.sss' string to total seconds."""
    time_str = time_str.strip('s')
    minutes, seconds = time_str.split('m')
    return float(minutes) * 60 + float(seconds)

times_v1 = [time_to_seconds(t) for t in raw_times_v1]
times_v2 = [time_to_seconds(t) for t in raw_times_v2]
times_v3 = [time_to_seconds(t) for t in raw_times_v3]

# =============================================================================
# 3. PLOTTING LOGIC (ACCESSIBLE VERSION)
# =============================================================================

# --- Setup for the grouped bar chart ---
x = np.arange(len(grid_sizes))
width = 0.25

# --- NEW: Define a color-blind friendly palette and patterns ---
# Using the Tableau Colorblind 10 palette
colors = ['#006BA4', '#FF800E', '#ABABAB'] # Blue, Orange, Grey
patterns = ['/', '\\', 'x'] # Slashes, back-slashes, crosses

# --- Create the figure and axes ---
fig, ax = plt.subplots(figsize=(10, 7)) # Increased height for better label spacing

# --- Create the bars for each version with color and hatch ---
rects1 = ax.bar(x - width, times_v1, width, label=version_labels[0],
                color=colors[0], hatch=patterns[0], edgecolor='black')
rects2 = ax.bar(x,         times_v2, width, label=version_labels[1],
                color=colors[1], hatch=patterns[1], edgecolor='black')
rects3 = ax.bar(x + width, times_v3, width, label=version_labels[2],
                color=colors[2], hatch=patterns[2], edgecolor='black')

# --- Add labels, title, and custom x-axis tick labels ---
ax.set_ylabel('Execution Time (seconds) - Logarithmic Scale', fontsize=12)
ax.set_title('Jacobi Solver Performance Comparison by Version and Grid Size', fontsize=14, pad=20)
ax.set_xticks(x)
ax.set_xticklabels(grid_sizes, fontsize=11)
ax.legend(title='Program Version')

# --- Set the y-axis to a logarithmic scale ---
ax.set_yscale('log')

# --- Add text labels on top of each bar ---
def autolabel(rects):
    """Attach a text label above each bar, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}s',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  # 4 points vertical offset for more space
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8, rotation=0) # Kept rotation at 0

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

# --- Final adjustments and saving the figure ---
fig.tight_layout()
plt.savefig('performance_chart_accessible.png', dpi=300)
plt.show()

print("Accessible figure 'performance_chart_accessible.png' has been created!")