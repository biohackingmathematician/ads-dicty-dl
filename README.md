# Dictyostelium Aggregation Center Prediction

This project implements models to predict aggregation centers of *Dictyostelium discoideum* cells from microscopy time-lapse movies.

**Author:** Agna Chan  
**Date:** November 24, 2025  
**Course:** Project 2, Applied Data Science, Columbia University

## Project Overview

The goal of this project is to predict where *Dictyostelium discoideum* cells will aggregate using early frames of time-lapse microscopy movies. The project uses multiple modeling approaches to predict either:
- The coordinates of eventual aggregation center(s), or
- A spatial probability map (heatmap) showing where aggregation is most likely to occur

## Dataset

The project uses three experimental datasets:
- `mixin_test44`: 2024-01-17 experiment
- `mixin_test57`: 2024-02-29 experiment  
- `mixin_test64`: 2024-04-04 experiment

Each experiment contains:
- **Original zarr files**: Used for training (higher resolution, more frames)
- **Subsampled zarr files**: Used for evaluation and robustness testing

## Project Structure

- `Agna-slime-mold.ipynb`: Main project notebook with models and experiments
- `run_experiments_minimal.py`: Standalone script for running all experiments (alternative to notebook)
- `results/`: Directory containing experiment results (JSON files)
- `data/`: Directory containing experimental data in zarr format

## Requirements

Install required packages:
```bash
pip install torch numpy scipy zarr scikit-learn
```

## Usage

### Option 1: Jupyter Notebook
1. Open `Agna-slime-mold.ipynb` in Jupyter Lab or Jupyter Notebook
2. Run cells sequentially (or "Run All")
3. Results will be saved to `results/` directory

### Option 2: Standalone Script
```bash
python3 run_experiments_minimal.py
```

The script will:
- Run all 3 experiments automatically
- Train 1 neural model (TinyCNN) + 2 baselines (GMM, LastFrame) per experiment
- Save results incrementally to `results/` directory
- Complete in ~2-3 minutes

## Key Features

- **Multiple Experiments**: Train and evaluate separately on each experimental dataset (mixin_test44, mixin_test57, mixin_test64)
- **Multiple Models**: 
  - TinyCNN: Lightweight neural network (~20K parameters)
  - GMM: Gaussian Mixture Model baseline (zero-shot)
  - LastFrame: Simple baseline using last input frame
- **Confidence Intervals**: Results reported as mean ± CI (95% confidence interval using t-distribution)
- **Time-based Split**: 70% train, 30% test split for each experiment
- **Fast Execution**: Optimized for CPU, completes in ~2-3 minutes

## Results

Results are saved as JSON files in the `results/` directory:
- `results_mixin_test44.json`: Results for experiment 1
- `results_mixin_test57.json`: Results for experiment 2
- `results_mixin_test64.json`: Results for experiment 3
- `all_results.json`: Complete summary of all experiments

Each result file contains mean, standard deviation, and 95% confidence intervals for each model.

## License

MIT License - see LICENSE file for details

