import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
import os

# ============================================
# CONFIGURATION
# ============================================
DATASET_PATH = "loan.csv"
TARGET_COLUMN = input("Enter target column name to remove (or press Enter to skip): ").strip()
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
X = numeric_df.to_numpy()

print(f"Dataset shape: {X.shape}")
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
    print(f"REDUCTION: {reduction_percentage}% (k={k} dimensions)")
    print(f"{'='*60}")
    
    nrp_distortions = []
    pmrp_distortions = []
    hrp_distortions = []
    
    for alpha in ALPHA_VALUES:
        # NRP
        R_nrp = create_nrp_matrix(n_features, k, RANDOM_STATE)
        X_nrp = X @ R_nrp
        distortion_nrp = compute_distortion(X, X_nrp)
        nrp_distortions.append(distortion_nrp)
        
        # PMRP
        R_pmrp = create_pmrp_matrix(n_features, k, RANDOM_STATE)
        X_pmrp = X @ R_pmrp
        distortion_pmrp = compute_distortion(X, X_pmrp)
        pmrp_distortions.append(distortion_pmrp)
        
        # HRP
        R_hrp = create_hrp_matrix(n_features, k, alpha, RANDOM_STATE)
        X_hrp = X @ R_hrp
        distortion_hrp = compute_distortion(X, X_hrp)
        hrp_distortions.append(distortion_hrp)
        
        print(f"Alpha={alpha:.1f} | NRP={distortion_nrp:.6f} | PMRP={distortion_pmrp:.6f} | HRP={distortion_hrp:.6f}")
    
    # Find best alpha for HRP
    min_idx = np.argmin(hrp_distortions)
    best_alpha = ALPHA_VALUES[min_idx]
    min_distortion = hrp_distortions[min_idx]
    
    best_alphas[reduction_percentage] = best_alpha
    best_distortions[reduction_percentage] = min_distortion
    
    print(f"\nBEST ALPHA for {reduction_percentage}% reduction: {best_alpha:.1f}")
    print(f"MINIMUM DISTORTION: {min_distortion:.6f}\n")
    
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
    plt.savefig(f'distortion_vs_alpha_{reduction_percentage}{DATASET_PATH}pct.png', dpi=300, bbox_inches='tight')
    plt.show()

# ============================================
# FINAL SUMMARY BAR GRAPH
# ============================================
print(f"\n{'='*60}")
print("FINAL SUMMARY")
print(f"{'='*60}")

for reduction_percentage in REDUCTION_PERCENTAGES:
    if reduction_percentage in best_alphas:
        print(f"Reduction: {reduction_percentage}% | Best Alpha: {best_alphas[reduction_percentage]:.1f} | Min Distortion: {best_distortions[reduction_percentage]:.6f}")

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