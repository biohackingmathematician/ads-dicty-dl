# Dictyostelium Aggregation Center Prediction

This project implements models to predict aggregation centers of *Dictyostelium discoideum* cells from microscopy time-lapse movies.

**Author:** Agna Chan  
**Date:** November 24, 2025  
**Course:** Project 2, Applied Data Science, Columbia University

## Project Overview

This project implements deep learning and statistical models to predict aggregation centers of *Dictyostelium discoideum* cells from early microscopy time-lapse frames. The goal is to determine how many consecutive frames (K) are needed to accurately predict where cells will aggregate.

**Key Features:**
- 2 neural models: SpatioTemporalCNN (3D CNN) and SimpleUNet
- 2 statistical baselines: GMM (Gaussian Mixture Model) and LastFrame
- 3 experiments evaluated separately with per-experiment training
- Multiple training seeds (42, 123, 456) for proper confidence intervals
- 95% CI reporting using t-distribution
- Comprehensive visualizations

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

See `requirements.txt` for full dependencies. Key packages:
- PyTorch >= 1.9
- NumPy
- SciPy
- Zarr
- scikit-learn
- Matplotlib
- Pandas

Install required packages:
```bash
pip install torch numpy scipy zarr scikit-learn matplotlib pandas
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
- Train neural models (SpatioTemporalCNN, SimpleUNet) + 2 baselines (GMM, LastFrame) per experiment
- Save results incrementally to `results/` directory
- Complete in ~45-60 minutes on CPU (includes training with multiple seeds)

## Models

**Neural Networks (require training):**
- `SpatioTemporalCNN`: 3D convolutional network that learns spatiotemporal features from K input frames
- `SimpleUNet`: Encoder-decoder architecture with skip connections for spatial feature extraction

**Statistical Baselines (zero-shot, no training):**
- `GMM`: Gaussian Mixture Model fitted to brightest pixels in averaged input frames
- `LastFrame`: Simple baseline using center-of-mass of the last input frame

## Methodology

**Data Split:** Time-based split using first 50% of frames as input, predicting final aggregation center from last 10 frames. Train/test split is 70%/30% of available windows.

**Evaluation Metric:** Euclidean distance (pixels) between predicted and true aggregation center.

**Statistical Reporting:** Mean and 95% confidence intervals computed using t-distribution across multiple training seeds and test samples.

**Ground Truth:** Aggregation center computed as center-of-mass of averaged final 10 frames.

## Results

### Summary Table (K=4 input frames)

| Experiment | Best Model | Mean Error (px) | 95% CI |
|------------|------------|-----------------|--------|
| mixin_test44 | SimpleUNet | 0.0003 | [0.0002, 0.0004] |
| mixin_test57 | SimpleUNet | 1.11 | [1.03, 1.18] |
| mixin_test64 | SimpleUNet | 0.17 | [0.14, 0.20] |

**Key Findings:**
- SimpleUNet achieves best performance across all experiments
- Neural models outperform GMM baseline by 16-30x on most experiments
- K=4 frames is sufficient for sub-pixel to few-pixel accuracy
- Models are robust to edge cases (corner aggregation in mixin_test44)

## Output Files

Results are saved to the `results/` directory:

**Data Files:**
- `all_results.json`: Complete results for all experiments
- `results_mixin_test44.json`: Individual experiment results
- `results_mixin_test57.json`
- `results_mixin_test64.json`
- `results_summary.csv`: Tabular summary of all results

**Visualizations:**
- `error_vs_k.png`: Model performance at K=4 input frames
- `model_comparison_bars.png`: Bar chart comparing models across experiments
- `training_curves.png`: Training and validation loss curves
- `qualitative_multiframe.png`: Input sequence and prediction visualization
- `viz/`: Per-experiment prediction overlay images

## License

MIT License - see LICENSE file for details

