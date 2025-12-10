import matplotlib.pyplot as plt
import csv
import sys

# Usage: python plot_results.py <path_to_csv_data>
# You will need to manually create a data.csv from your log files 
# format: Implementation,Time_Seconds

def plot_performance(data_file):
    labels = []
    times = []
    
    try:
        with open(data_file, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                labels.append(row[0])
                times.append(float(row[1]))
    except Exception as e:
        print(f"Error reading file: {e}")
        return

    plt.figure(figsize=(10, 6))
    bars = plt.bar(labels, times, color=['gray', 'green', 'blue'])
    
    plt.ylabel('Execution Time (s)')
    plt.title('Laplace Solver Performance (4096^2, 10k iter)')
    
    # Add value labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, round(yval, 2), va='bottom', ha='center')

    plt.savefig('../results/plots/performance_comparison.png')
    print("Plot saved to results/plots/performance_comparison.png")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_performance(sys.argv[1])
    else:
        print("Please provide a CSV file path.")