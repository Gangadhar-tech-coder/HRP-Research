import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = input("Enter dataset path (default: mnist_train.csv): ").strip()
if not DATASET_PATH:
    DATASET_PATH = "datasets/mnist_train.csv"

TARGET_COLUMN = input("Enter target column name to remove (or press Enter to skip): ").strip()

# NEW: Ask user for number of samples to use
print("\n--- Sample Selection ---")
full_dataset_info = input("Do you want to use a subset of samples? (y/n): ").strip().lower()

REDUCTION_PERCENTAGES = [70, 50, 30, 10]
ALPHA_VALUES = np.arange(0.0, 1.1, 0.1)
RANDOM_STATE = 42

# ============================================
# LOAD AND PREPARE DATASET
# ============================================
print(f"\nLoading dataset from: {DATASET_PATH}")
df = pd.read_csv(DATASET_PATH)

if TARGET_COLUMN and TARGET_COLUMN in df.columns:
    print(f"Removing target column: {TARGET_COLUMN}")
    df = df.drop(columns=[TARGET_COLUMN])

numeric_df = df.select_dtypes(include=[np.number])


# NEW: Sample selection logic
print(f"\nFull dataset shape: {numeric_df.shape}")
print(f"Total samples available: {numeric_df.shape[0]}, Features: {numeric_df.shape[1]}")

if full_dataset_info == 'y':
    n_samples_to_use = int(input(f"Enter number of samples to use (max {numeric_df.shape[0]}): ").strip())
    if n_samples_to_use > numeric_df.shape[0]:
        print(f"Warning: Requested {n_samples_to_use} samples but only {numeric_df.shape[0]} available. Using all samples.")
        n_samples_to_use = numeric_df.shape[0]
    
    # Randomly sample n_samples_to_use rows
    np.random.seed(RANDOM_STATE)
    sampled_indices = np.random.choice(numeric_df.shape[0], size=n_samples_to_use, replace=False)
    numeric_df = numeric_df.iloc[sampled_indices].reset_index(drop=True)
    print(f"\nUsing {n_samples_to_use} randomly sampled instances")
else:
    print("\nUsing all available samples")

X = numeric_df.to_numpy()

print(f"\nFinal dataset shape: {X.shape}")
print(f"Samples: {X.shape[0]}, Features: {X.shape[1]}\n")

n_samples, n_features = X.shape

# ============================================
# RANDOM PROJECTION IMPLEMENTATIONS
# ============================================
def create_nrp_matrix(n_features, n_components, random_state):
    np.random.seed(random_state)
    return np.random.randn(n_features, n_components) / np.sqrt(n_components)

def create_pmrp_matrix(n_features, n_components, random_state):
    np.random.seed(random_state)
    return np.random.choice([-1, 1], size=(n_features, n_components)) / np.sqrt(n_components)

def create_hrp_matrix(n_features, n_components, alpha, random_state):
    R1 = create_nrp_matrix(n_features, n_components, random_state)
    R2 = create_pmrp_matrix(n_features, n_components, random_state + 1)
    return alpha * R1 + (1 - alpha) * R2

def compute_distortion(X_original, X_projected, n_samples_dist=5000):
    """Compute distortion using sampled distances to reduce memory usage."""
    n_samples = X_original.shape[0]
    ratios = []
    
    # Sample random pairs of points
    n_pairs = min(n_samples_dist, n_samples * (n_samples - 1) // 2)
    
    np.random.seed(RANDOM_STATE)  # For reproducibility
    for _ in range(n_pairs):
        i, j = np.random.choice(n_samples, 2, replace=False)
        
        d_orig = np.linalg.norm(X_original[i] - X_original[j])
        d_proj = np.linalg.norm(X_projected[i] - X_projected[j])
        
        if d_orig > 1e-10:
            ratio = np.abs(d_proj / d_orig - 1)
            ratios.append(ratio)
    
    return np.mean(ratios) if ratios else 0.0

# ============================================
# MAIN EXPERIMENT LOOP
# ============================================
best_alphas = {}
best_distortions = {}

for reduction_percentage in REDUCTION_PERCENTAGES:
    k = int(n_features * reduction_percentage / 100)
    
    if k < 1:
        print(f"Skipping reduction {reduction_percentage}% (k={k} too small)")
        continue
    
    print(f"\n{'='*60}")
    print(f"Reduction: {reduction_percentage}% | Reduced Dimension: {k}")
    print(f"{'='*60}")
    print(f"{'Alpha':<8} {'NRP Distortion':<18} {'PMRP Distortion':<18} {'HRP Distortion':<18}")
    print(f"{'-'*60}")
    
    # OPTIMIZED: Calculate NRP and PMRP only ONCE per reduction
    print("Calculating NRP projection...")
    R_nrp = create_nrp_matrix(n_features, k, RANDOM_STATE)
    X_nrp = X @ R_nrp
    distortion_nrp = compute_distortion(X, X_nrp)
    
    print("Calculating PMRP projection...")
    R_pmrp = create_pmrp_matrix(n_features, k, RANDOM_STATE)
    X_pmrp = X @ R_pmrp
    distortion_pmrp = compute_distortion(X, X_pmrp)
    
    print("Calculating HRP for all alpha values...\n")
    
    # Store distortions for plotting
    nrp_distortions = []
    pmrp_distortions = []
    hrp_distortions = []
    
    # OPTIMIZED: Only calculate HRP for each alpha
    for alpha in ALPHA_VALUES:
        # NRP and PMRP distortions are constant across all alphas
        nrp_distortions.append(distortion_nrp)
        pmrp_distortions.append(distortion_pmrp)
        
        # HRP - Calculate for each alpha value
        R_hrp = create_hrp_matrix(n_features, k, alpha, RANDOM_STATE)
        X_hrp = X @ R_hrp
        distortion_hrp = compute_distortion(X, X_hrp)
        hrp_distortions.append(distortion_hrp)
        
        print(f"{alpha:<8.1f} {distortion_nrp:<18.8f} {distortion_pmrp:<18.8f} {distortion_hrp:<18.8f}")
    
    # Find best alpha for HRP
    min_idx = np.argmin(hrp_distortions)
    best_alpha = ALPHA_VALUES[min_idx]
    min_distortion = hrp_distortions[min_idx]
    
    best_alphas[reduction_percentage] = best_alpha
    best_distortions[reduction_percentage] = min_distortion
    
    print(f"\nBest Alpha for {reduction_percentage}% Reduction: {best_alpha:.1f}")
    print(f"Minimum HRP Distortion: {min_distortion:.8f}\n")
    
    # Plot distortion vs alpha
    plt.figure(figsize=(10, 6))
    plt.plot(ALPHA_VALUES, nrp_distortions, 'o-', label='NRP', linewidth=2, markersize=8)
    plt.plot(ALPHA_VALUES, pmrp_distortions, 's-', label='PMRP', linewidth=2, markersize=8)
    plt.plot(ALPHA_VALUES, hrp_distortions, '^-', label='HRP', linewidth=2, markersize=8)
    plt.axvline(best_alpha, color='red', linestyle='--', linewidth=2, alpha=0.7, label=f'Best α={best_alpha:.1f}')
    
    plt.xlabel('Alpha (α)', fontsize=14)
    plt.ylabel('Distance Distortion', fontsize=14)
    plt.title(f'Distance Distortion vs Alpha (Reduction = {reduction_percentage}%)', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'distortion_vs_alpha_{reduction_percentage}pct.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# FINAL SUMMARY TABLE
# ============================================
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")
print(f"{'Reduction %':<15} {'Best Alpha':<15} {'Minimum Distortion':<20}")
print(f"{'-'*60}")

for reduction_percentage in REDUCTION_PERCENTAGES:
    if reduction_percentage in best_alphas:
        print(f"{reduction_percentage:<15} {best_alphas[reduction_percentage]:<15.1f} {best_distortions[reduction_percentage]:<20.8f}")

print(f"{'='*60}")

# ============================================
# FINAL SUMMARY BAR GRAPH
# ============================================
reductions = list(best_alphas.keys())
alphas = list(best_alphas.values())

plt.figure(figsize=(10, 6))
bars = plt.bar(reductions, alphas, color='steelblue', edgecolor='black', linewidth=1.5, alpha=0.8)

for i, (reduction, alpha) in enumerate(zip(reductions, alphas)):
    plt.text(reduction, alpha + 0.02, f'{alpha:.1f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

plt.xlabel('Dimensionality Reduction (%)', fontsize=14)
plt.ylabel('Best Alpha (α)', fontsize=14)
plt.title('Best Alpha vs Dimensionality Reduction', fontsize=16, fontweight='bold')
plt.ylim(0, 1.2)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('best_alpha_vs_reduction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\nExperiment completed successfully!")