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
- `Agna-DL-pytorch.ipynb`: PyTorch learning exercises (prerequisites)
- `data/`: Directory containing experimental data in zarr format

## Requirements

Install required packages:
```bash
pip install torch numpy matplotlib zarr numcodecs tifffile h5py scikit-image tqdm
```

## Usage

1. Open `Agna-slime-mold.ipynb` in Jupyter
2. Set `EXPERIMENT_NAME` to choose which experiment to analyze
3. Run cells sequentially to:
   - Load and visualize data
   - Train models on original data
   - Evaluate on subsampled data
   - Run automated experiments across all datasets

## Key Features

- **Multiple Experiments**: Train and evaluate separately on each experimental dataset
- **Robustness Testing**: Evaluate models trained on original data using subsampled data
- **Automated Runs**: Run multiple training runs with different seeds to compute confidence intervals
- **Multiple Models**: Baseline CNN plus additional modeling approaches

## Results

Results are reported as mean ± CI (confidence interval) from multiple training runs on the same data, following standard ML practice.

## License

MIT License - see LICENSE file for details

