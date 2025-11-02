#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=13_plot_gen_out_%j.out
#SBATCH --error=14_plot_gen_err_%j.err
#SBATCH --time=00:05:00
#SBATCH --job-name=GeneratePlots

echo "====================================================================================="
echo "                         JOB: GENERATE PLOTS ONLY                                  "
echo "====================================================================================="
echo ""

# 1. LOAD MODULES
echo "--- Loading modules ---" >&2
# Load GCC (good practice for C++ libraries) and Miniconda
module add gcc/12.2.1
module add miniconda/3
echo "--- Modules loaded ---" >&2
echo ""

# 2. SETUP PYTHON ENVIRONMENT
echo "--- Setting up private Python environment ---" >&2
# Define a path for your private conda environment
export MY_PYTHON_ENV=~/my-openmp-env

# Check if the environment already exists
if [ ! -d "$MY_PYTHON_ENV" ]; then
    echo "--- Python environment not found. Creating new one at $MY_PYTHON_ENV ---" >&2
    # Create the env in your home dir (-p) and install pandas/matplotlib non-interactively (-y)
    # We install both pandas and matplotlib for the plotting script.
    conda create -p "$MY_PYTHON_ENV" pandas matplotlib -y
    
    if [ $? -ne 0 ]; then
        echo "Error: Failed to create the conda environment. Exiting." >&2
        exit 1
    fi
    echo "--- Environment created successfully. ---" >&2
else
    echo "--- Private conda environment already exists. ---" >&2
fi

# Activate the private environment
echo "--- Activating private conda environment ---" >&2
source activate "$MY_PYTHON_ENV"
echo "--- Python environment is ready ---" >&2
echo ""

# 3. RUN PYTHON PLOTTING SCRIPT
echo "--- Generating plots from 07_...csv and 08_...csv ---" >&2
# Check if the input CSV files exist
if [ ! -f "07_speedup_data.csv" ] || [ ! -f "08_bottleneck_data.csv" ]; then
    echo "Error: Input CSV files (07_...csv, 08_...csv) not found." >&2
    echo "Please run the main '01_run_fire_sim.slurm' script to generate them." >&2
    exit 1
fi

# Run the plotting script
python3 04_generate_plots.py

if [ $? -eq 0 ]; then
    echo "--- Plot generation finished successfully. ---" >&2
    echo "Generated 09_speedup_analysis.png and 10_bottleneck_analysis.png"
else
    echo "Error: Python plotting script failed." >&2
    exit 1
fi
echo ""

echo "====================================================================================="
echo "                         SLURM JOB COMPLETE                                          "
echo "====================================================================================="