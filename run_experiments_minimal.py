#!/usr/bin/env python3
"""
Dictyostelium Aggregation Prediction - Minimal Version
Standalone script version for local execution with progress monitoring
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scipy import stats
import zarr
import gc
# Configuration
DATA_ROOT = "data"
RESULTS_DIR = "results"
K = 4  # Number of input frames
EPOCHS = 10  # Reduced for speed
BATCH_SIZE = 4  # Small for CPU
LR = 1e-3
DEVICE = "cpu"  # Force CPU to avoid CUDA issues
SEED = 42

EXPERIMENTS = {
    "mixin_test44": "data/mixin_test44/2024-01-17_ERH_23hr_ERH Red FarRed.zarr",
    "mixin_test57": "data/mixin_test57/2024-02-29_mixin57_overnight_25um_ERH_Red_FarRed_25.zarr",
    "mixin_test64": "data/mixin_test64/ERH_2024-04-04_mixin64_wellC5_10x_overnight_ERH Red FarRed_1.zarr",
}

np.random.seed(SEED)
torch.manual_seed(SEED)

print(f"Configuration:")
print(f"  K={K}, Epochs={EPOCHS}, Batch={BATCH_SIZE}, Device={DEVICE}")
def load_movie(path):
    """Load and normalize zarr movie."""
    if not os.path.exists(path):
        print(f"  ERROR: {path} not found")
        return None
    
    # Load zarr array (works for both directory and file formats)
    z = zarr.open(path, mode='r')
    data = np.array(z)
    
    # Handle multi-channel: (T, C, H, W) -> (T, H, W)
    if data.ndim == 4:
        data = data[:, 0]  # Take first channel
    elif data.ndim == 5:
        data = data[:, 0, 0]  # Take first channel, first slice
    
    # Normalize
    data = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data.astype(np.float32)
def get_final_aggregation_center(movie, final_window=10):
    """Compute the final aggregation center from the last frames of the movie."""
    if len(movie) < final_window:
        final_window = len(movie)
    
    # Use last frames to find where cells actually aggregated
    final_frames = movie[-final_window:]
    final_avg = final_frames.mean(axis=0)  # Average of last frames
    
    # Compute center of mass
    H, W = final_avg.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    total = final_avg.sum() + 1e-8
    cy = (ys * final_avg).sum() / total
    cx = (xs * final_avg).sum() / total
    return np.array([cy, cx], dtype=np.float32)

class SimpleDataset(Dataset):
    """Dataset for predicting final aggregation center from early frames.
    
    Input: Early frames (from first 50% of movie)
    Target: Final aggregation center (from last frames of movie)
    """
    def __init__(self, movie, k=4, final_center=None):
        self.movie = movie
        self.k = k
        self.final_center = final_center if final_center is not None else get_final_aggregation_center(movie)
        
        # Only use early frames (first 50% of movie) for training
        self.max_start = len(movie) // 2
    
    def __len__(self):
        # Create samples from early portion of movie
        return max(1, self.max_start - self.k)
    
    def __getitem__(self, i):
        # Input: Early frames (from first half of movie)
        x = torch.from_numpy(self.movie[i:i+self.k])  # (K, H, W)
        # Target: Final aggregation center (same for all samples from this movie)
        y = torch.from_numpy(self.final_center)  # (2,) - [cy, cx] coordinates
        return x, y
class TinyCNN(nn.Module):
    """Minimal CNN that predicts final aggregation center coordinates from early frames."""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(K, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 8, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)  # Global average pooling
        self.fc = nn.Linear(8, 2)  # Output: (cy, cx) coordinates
    
    def forward(self, x):
        # x: (B, K, H, W)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # (B, 8, 1, 1)
        x = x.view(x.size(0), -1)  # (B, 8)
        x = self.fc(x)  # (B, 2) - [cy, cx]
        return x
def center_of_mass(img):
    """Get center of mass of 2D image."""
    img = np.squeeze(img)
    if img.ndim != 2:
        return (img.shape[-2]/2, img.shape[-1]/2)
    
    H, W = img.shape
    ys, xs = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    total = img.sum() + 1e-8
    cy = (ys * img).sum() / total
    cx = (xs * img).sum() / total
    return float(cy), float(cx)
def gmm_predict_early_frames(frames):
    """GMM baseline - predict final center from early frames."""
    try:
        from sklearn.mixture import GaussianMixture
    except ImportError:
        # Fallback: use center of mass of averaged early frames
        if isinstance(frames, torch.Tensor):
            frames = frames.numpy()
        frames = np.squeeze(frames)
        if frames.ndim == 3:
            avg = frames.mean(axis=0)
        else:
            avg = frames
        return center_of_mass(avg)
    
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    frames = np.squeeze(frames)
    
    # Average early frames to get initial pattern
    if frames.ndim == 3:
        avg = frames.mean(axis=0)
    else:
        avg = frames
    
    # Find brightest regions (where cells are clustering)
    H, W = avg.shape
    thr = np.percentile(avg, 95)
    ys, xs = np.where(avg >= thr)
    
    if len(ys) < 2:
        return center_of_mass(avg)
    
    # Fit GMM to bright pixels
    coords = np.stack([ys, xs], axis=1)
    gmm = GaussianMixture(n_components=1, random_state=42)
    gmm.fit(coords)
    cy, cx = gmm.means_[0]
    return float(cy), float(cx)
def lastframe_predict_early(frames):
    """LastFrame baseline - use center of last early frame.
    This is a simple baseline that assumes the center doesn't move much."""
    if isinstance(frames, torch.Tensor):
        frames = frames.numpy()
    frames = np.squeeze(frames)
    # Use the last of the early frames (not the actual last frame of movie)
    last_early = frames[-1] if frames.ndim == 3 else frames
    return center_of_mass(last_early)
def euclidean_error(pred, true):
    """Compute Euclidean distance between predicted and true center."""
    return np.sqrt((pred[0]-true[0])**2 + (pred[1]-true[1])**2)
def compute_ci(errors):
    """Compute mean and 95% CI using t-distribution."""
    errors = np.array(errors)
    n = len(errors)
    mean = errors.mean()
    std = errors.std(ddof=1) if n > 1 else 0.0
    
    if n > 1 and std > 1e-10:  # Only compute CI if there's variance
        se = std / np.sqrt(n)
        ci_low, ci_high = stats.t.interval(0.95, df=n-1, loc=mean, scale=se)
        ci_low, ci_high = float(ci_low), float(ci_high)
    else:
        # For zero variance or single sample, CI equals mean
        ci_low, ci_high = float(mean), float(mean)
    
    return {"mean": float(mean), "std": float(std), 
            "ci_low": ci_low, "ci_high": ci_high, "n": n}
def train_model(train_loader, test_loader):
    """Train TinyCNN to predict final aggregation center coordinates."""
    model = TinyCNN().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss()  # L2 loss on coordinates
    
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)  # (B, 2) - predicted [cy, cx]
            loss = criterion(pred, yb)  # yb is (B, 2) - true [cy, cx]
            loss.backward()
            opt.step()
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                    val_loss += criterion(model(xb), yb).item()
            print(f"    Epoch {epoch+1}: val_loss={val_loss/len(test_loader):.4f}")
    
    return model
def evaluate_model(model, loader, final_center, model_type='cnn'):
    """Evaluate any model and return center errors.
    
    Compares predicted final center (from early frames) vs actual final center.
    """
    errors = []
    if model_type == 'cnn':
        model.eval()
    
    with torch.no_grad():
        for xb, yb in loader:
            for i in range(xb.shape[0]):
                x_i = xb[i]  # Early frames: (K, H, W)
                # yb contains the true final center (same for all samples)
                true_center = final_center  # (2,) [cy, cx]
                
                if model_type == 'cnn':
                    # Model directly outputs coordinates
                    pred_coords = model(x_i.unsqueeze(0).to(DEVICE)).cpu().numpy()[0]
                    pred_center = (pred_coords[0], pred_coords[1])
                elif model_type == 'gmm':
                    pred_center = gmm_predict_early_frames(x_i)
                else:  # lastframe
                    pred_center = lastframe_predict_early(x_i)
                
                errors.append(euclidean_error(pred_center, true_center))
    
    return errors
def run_single_experiment(name, path):
    """Run one experiment."""
    print(f"\n{'='*50}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*50}")
    
    # Load
    movie = load_movie(path)
    if movie is None:
        return None
    print(f"  Loaded: {movie.shape}")
    
    # Compute final aggregation center (ground truth)
    final_center = get_final_aggregation_center(movie)
    print(f"  Final aggregation center (ground truth): ({final_center[0]:.1f}, {final_center[1]:.1f})")
    
    # Dataset: Use early frames (first 50%) to predict final center
    ds = SimpleDataset(movie, K, final_center=final_center)
    split = int(len(ds) * 0.7)
    train_ds = torch.utils.data.Subset(ds, range(split))
    test_ds = torch.utils.data.Subset(ds, range(split, len(ds)))
    print(f"  Using early frames (first {ds.max_start}/{len(movie)} frames) to predict final center")
    print(f"  Train samples: {len(train_ds)}, Test samples: {len(test_ds)}")
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    print(f"  Train: {len(train_ds)}, Test: {len(test_ds)}")
    
    results = {}
    
    # 1. TinyCNN
    print("\n  [1/3] Training TinyCNN...")
    t0 = time.time()
    model = train_model(train_loader, test_loader)
    cnn_errors = evaluate_model(model, test_loader, final_center, 'cnn')
    results['TinyCNN'] = compute_ci(cnn_errors)
    print(f"    Done in {time.time()-t0:.1f}s: {results['TinyCNN']['mean']:.2f} ± {results['TinyCNN']['std']:.2f} px")
    
    # 2. GMM (instant)
    print("  [2/3] GMM baseline...")
    gmm_errors = evaluate_model(None, test_loader, final_center, 'gmm')
    results['GMM'] = compute_ci(gmm_errors)
    print(f"    {results['GMM']['mean']:.2f} ± {results['GMM']['std']:.2f} px")
    
    # 3. LastFrame (instant)
    print("  [3/3] LastFrame baseline...")
    lf_errors = evaluate_model(None, test_loader, final_center, 'lastframe')
    results['LastFrame'] = compute_ci(lf_errors)
    print(f"    {results['LastFrame']['mean']:.2f} ± {results['LastFrame']['std']:.2f} px")
    
    # Cleanup
    del model, movie
    gc.collect()
    
    return results
print("="*60)
print("DICTYOSTELIUM PREDICTION - MINIMAL VERSION")
print("="*60)
print(f"K={K}, Epochs={EPOCHS}, Batch={BATCH_SIZE}, Device={DEVICE}")

os.makedirs(RESULTS_DIR, exist_ok=True)
all_results = {}

total_experiments = len(EXPERIMENTS)
for idx, (name, path) in enumerate(EXPERIMENTS.items(), 1):
    print(f"\n{'='*60}")
    print(f"PROGRESS: Experiment {idx}/{total_experiments}")
    print(f"{'='*60}")
    results = run_single_experiment(name, path)
    if results:
        all_results[name] = results
        # Save after each experiment
        with open(f"{RESULTS_DIR}/results_{name}.json", 'w') as f:
            json.dump(results, f, indent=2)
        print(f"  ✓ Saved: {RESULTS_DIR}/results_{name}.json")
        print(f"  ✓ Completed: {idx}/{total_experiments} experiments ({idx/total_experiments*100:.1f}%)")
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)
for exp, res in all_results.items():
    print(f"\n{exp}:")
    for model, stats in res.items():
        print(f"  {model}: {stats['mean']:.2f} ± {stats['std']:.2f} px "
              f"(95% CI: [{stats['ci_low']:.2f}, {stats['ci_high']:.2f}])")
# Save all results
with open(f"{RESULTS_DIR}/all_results.json", 'w') as f:
    json.dump(all_results, f, indent=2)

print(f"\nResults saved to {RESULTS_DIR}/")
print("\n✓ All experiments completed!")
