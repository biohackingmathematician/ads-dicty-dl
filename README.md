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
- `Agna-DL-pytorch.ipynb`: PyTorch Basics Exercises Completed - Deep Learning & Data Analysis
- `Agna-slime-mold.ipynb`: Main project notebook with models and experiments
- `run_experiments_minimal.py`: Standalone script for running all experiments (alternative to notebook)
- `results/`: Directory containing experiment results and visualizations
  - `results_mixin_test44.json`, `results_mixin_test57.json`, `results_mixin_test64.json`: Per-experiment results (JSON)
  - `all_results.json`: Complete summary of all experiments (JSON)
  - `results_summary.csv`: Summary table of all results (CSV)
  - `error_vs_k.png`: Visualization showing model performance at K=4 input frames
  - `model_comparison_bars.png`: Bar chart comparing all models across experiments
  - `training_curves.png`: Training and validation loss curves for neural models
  - `qualitative_multiframe.png`: Multi-panel visualization showing input frames and prediction overlay
  - `viz/`: Subdirectory containing per-experiment prediction overlays
    - `mixin_test44/`, `mixin_test57/`, `mixin_test64/`: Per-experiment visualization directories
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
  - **SpatioTemporalCNN**: 3D CNN with temporal pooling for spatiotemporal feature learning (~50K parameters)
  - **SimpleUNet**: U-Net style encoder-decoder with skip connections (~200K parameters)
  - **GMM**: Gaussian Mixture Model baseline (zero-shot, no training)
  - **LastFrame**: Simple baseline using center of mass of last input frame
- **Confidence Intervals**: Results reported as mean ± CI (95% confidence interval using t-distribution) with multiple training seeds
- **Time-based Split**: 70% train, 30% test split for each experiment
- **Visualizations**: Comprehensive plots including error analysis, model comparisons, training curves, and qualitative predictions
- **Fast Execution**: Optimized for CPU execution

## Results

Results are saved in the `results/` directory:

### Data Files
- `results_mixin_test44.json`: Results for experiment 1 (JSON)
- `results_mixin_test57.json`: Results for experiment 2 (JSON)
- `results_mixin_test64.json`: Results for experiment 3 (JSON)
- `all_results.json`: Complete summary of all experiments (JSON)
- `results_summary.csv`: Summary table with all results in CSV format

Each result file contains mean, standard deviation, and 95% confidence intervals for each model.

### Visualization Files
- `error_vs_k.png`: Model performance comparison at K=4 input frames
- `model_comparison_bars.png`: Grouped bar chart comparing all models (SpatioTemporalCNN, SimpleUNet, GMM, LastFrame) across experiments
- `training_curves.png`: Training and validation loss curves for SpatioTemporalCNN and SimpleUNet
- `qualitative_multiframe.png`: Multi-panel figure showing input sequence (4 frames) and final frame with predicted vs true aggregation center overlay
- `viz/`: Per-experiment prediction overlay visualizations
  - `mixin_test44/spatiotemporal_cnn.png`: Prediction overlay for experiment 1
  - `mixin_test57/spatiotemporal_cnn.png`: Prediction overlay for experiment 2
  - `mixin_test64/spatiotemporal_cnn.png`: Prediction overlay for experiment 3

## License

MIT License - see LICENSE file for details

