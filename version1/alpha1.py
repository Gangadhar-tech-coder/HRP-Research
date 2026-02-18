"""
Hybrid Random Projection (HRP) Research Program
Study of alpha parameter's effect on distance distortion
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset_path': 'mnist_train.csv',
    'target_column': 'label',
    'n_samples': 500,
    'n_features': None,
    'reduction_ratio': 0.5,
    'alpha_min': 0.1,
    'alpha_max': 1.0,
    'alpha_step': 0.1,
    'random_seed': 42,
    'normalize': True,
    'output_plot': 'hrp_alpha_sensitivity.png'
}


# ============================================================================
# DATA LOADING AND PREPROCESSING
# ============================================================================

def load_and_preprocess_data(config):
    """
    Load dataset from CSV and preprocess it.
    
    Parameters:
        config: Configuration dictionary
    
    Returns:
        data: Preprocessed numpy array
        original_shape: Tuple of (n_samples, n_features)
    """
    print(f"Loading dataset from: {config['dataset_path']}")
    
    # Load CSV
    df = pd.read_csv(config['dataset_path'])
    
    # Remove target column if specified
    if config['target_column'] and config['target_column'] in df.columns:
        df = df.drop(columns=[config['target_column']])
        print(f"Removed target column: {config['target_column']}")
    
    # Keep only numeric columns
    df = df.select_dtypes(include=[np.number])
    
    # Select number of features if specified
    if config['n_features'] is not None:
        n_cols = min(config['n_features'], df.shape[1])
        df = df.iloc[:, :n_cols]
    
    # Select number of samples
    n_rows = min(config['n_samples'], df.shape[0])
    df = df.iloc[:n_rows, :]
    
    # Convert to numpy array
    data = df.values
    
    # Normalize if specified
    if config['normalize']:
        scaler = StandardScaler()
        data = scaler.fit_transform(data)
        print("Data normalized using StandardScaler")
    
    print(f"Dataset shape: {data.shape}")
    
    return data, data.shape


# ============================================================================
# RANDOM PROJECTION MATRICES
# ============================================================================

def generate_normal_projection_matrix(d, k, seed=None):
    """
    Generate Normal Random Projection matrix (Gaussian distribution).
    
    Parameters:
        d: Original dimension
        k: Reduced dimension
        seed: Random seed
    
    Returns:
        R: Projection matrix of shape (d, k)
    """
    if seed is not None:
        np.random.seed(seed)
    
    R = np.random.normal(0, 1, size=(d, k))
    R = R / np.sqrt(k)  # Normalize
    
    return R


def generate_plusminus_projection_matrix(d, k, seed=None):
    """
    Generate Plus-Minus One Random Projection matrix.
    
    Parameters:
        d: Original dimension
        k: Reduced dimension
        seed: Random seed
    
    Returns:
        R: Projection matrix of shape (d, k)
    """
    if seed is not None:
        np.random.seed(seed)
    
    R = np.random.choice([-1, 1], size=(d, k))
    R = R / np.sqrt(k)  # Normalize
    
    return R


def generate_hybrid_projection_matrix(d, k, alpha, seed=None):
    """
    Generate Hybrid Random Projection matrix.
    H = alpha * R_normal + (1 - alpha) * R_pm
    
    Parameters:
        d: Original dimension
        k: Reduced dimension
        alpha: Blending parameter (0 to 1)
        seed: Random seed
    
    Returns:
        H: Hybrid projection matrix of shape (d, k)
    """
    R_normal = generate_normal_projection_matrix(d, k, seed)
    R_pm = generate_plusminus_projection_matrix(d, k, seed + 1 if seed else None)
    
    H = alpha * R_normal + (1 - alpha) * R_pm
    
    return H


# ============================================================================
# DISTANCE DISTORTION COMPUTATION
# ============================================================================

def compute_pairwise_distances(data):
    """
    Compute pairwise Euclidean distances.
    
    Parameters:
        data: Data matrix of shape (n_samples, n_features)
    
    Returns:
        distances: Pairwise distances (condensed form)
    """
    distances = pdist(data, metric='euclidean')
    return distances


def compute_distance_distortion(original_distances, projected_distances):
    """
    Compute distance distortion metric.
    
    Distortion = mean(|(d_projected / d_original) - 1|)
    
    Parameters:
        original_distances: Original pairwise distances
        projected_distances: Projected pairwise distances
    
    Returns:
        distortion: Mean absolute relative error
    """
    # Avoid division by zero
    mask = original_distances > 1e-10
    
    if np.sum(mask) == 0:
        return 0.0
    
    ratio = projected_distances[mask] / original_distances[mask]
    distortion = np.mean(np.abs(ratio - 1))
    
    return distortion


def project_and_compute_distortion(data, projection_matrix, original_distances):
    """
    Project data and compute distortion.
    
    Parameters:
        data: Original data
        projection_matrix: Projection matrix
        original_distances: Original pairwise distances
    
    Returns:
        distortion: Distance distortion value
    """
    # Project data
    projected_data = np.dot(data, projection_matrix)
    
    # Compute projected distances
    projected_distances = compute_pairwise_distances(projected_data)
    
    # Compute distortion
    distortion = compute_distance_distortion(original_distances, projected_distances)
    
    return distortion


# ============================================================================
# ALPHA SENSITIVITY STUDY
# ============================================================================

def run_alpha_sensitivity_study(data, k, config):
    """
    Study effect of alpha on distance distortion.
    
    Parameters:
        data: Preprocessed data
        k: Reduced dimension
        config: Configuration dictionary
    
    Returns:
        results: Dictionary containing alpha values and distortions
    """
    n_samples, d = data.shape
    
    # Compute original pairwise distances
    print("\nComputing original pairwise distances...")
    original_distances = compute_pairwise_distances(data)
    
    # Generate alpha values
    alpha_values = np.arange(
        config['alpha_min'],
        config['alpha_max'] + config['alpha_step'] / 2,
        config['alpha_step']
    )
    
    # Initialize result storage
    nrp_distortions = []
    pmrp_distortions = []
    hrp_distortions = []
    
    print("\n" + "="*80)
    print("ALPHA SENSITIVITY STUDY")
    print("="*80)
    print(f"{'Alpha':<10} {'NRP Distortion':<20} {'PMRP Distortion':<20} {'HRP Distortion':<20}")
    print("-"*80)
    
    # Run study for each alpha value
    for alpha in alpha_values:
        seed = config['random_seed']
        
        # Normal Random Projection
        R_normal = generate_normal_projection_matrix(d, k, seed)
        nrp_distortion = project_and_compute_distortion(data, R_normal, original_distances)
        nrp_distortions.append(nrp_distortion)
        
        # Plus-Minus One Random Projection
        R_pm = generate_plusminus_projection_matrix(d, k, seed + 1)
        pmrp_distortion = project_and_compute_distortion(data, R_pm, original_distances)
        pmrp_distortions.append(pmrp_distortion)
        
        # Hybrid Random Projection
        H = generate_hybrid_projection_matrix(d, k, alpha, seed)
        hrp_distortion = project_and_compute_distortion(data, H, original_distances)
        hrp_distortions.append(hrp_distortion)
        
        # Print results
        print(f"{alpha:<10.1f} {nrp_distortion:<20.6f} {pmrp_distortion:<20.6f} {hrp_distortion:<20.6f}")
    
    print("="*80)
    
    results = {
        'alpha': alpha_values,
        'nrp': np.array(nrp_distortions),
        'pmrp': np.array(pmrp_distortions),
        'hrp': np.array(hrp_distortions)
    }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_alpha_sensitivity(results, config):
    """
    Plot alpha vs distortion for all methods.
    
    Parameters:
        results: Dictionary containing alpha values and distortions
        config: Configuration dictionary
    """
    plt.figure(figsize=(10, 6))
    
    # Plot curves
    plt.plot(results['alpha'], results['nrp'], 'b-o', label='Normal Random Projection (NRP)', linewidth=2, markersize=6)
    plt.plot(results['alpha'], results['pmrp'], 'r-s', label='Plus-Minus Random Projection (PMRP)', linewidth=2, markersize=6)
    plt.plot(results['alpha'], results['hrp'], 'g-^', label='Hybrid Random Projection (HRP)', linewidth=2, markersize=6)
    
    # Formatting
    plt.xlabel('Alpha (α)', fontsize=12)
    plt.ylabel('Distance Distortion', fontsize=12)
    plt.title('Effect of Alpha on Distance Distortion in Random Projections', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot
    output_path = config['output_plot']
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Display plot
    plt.show()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """
    Main function to execute the HRP research program.
    """
    print("="*80)
    print("HYBRID RANDOM PROJECTION (HRP) RESEARCH PROGRAM")
    print("="*80)
    
    # Load and preprocess data
    data, original_shape = load_and_preprocess_data(CONFIG)
    
    n_samples, d = data.shape
    
    # Compute reduced dimension
    k = int(d * CONFIG['reduction_ratio'])
    print(f"\nOriginal dimension: {d}")
    print(f"Reduced dimension: {k} ({CONFIG['reduction_ratio']*100:.0f}% of original)")
    
    # Run alpha sensitivity study
    results = run_alpha_sensitivity_study(data, k, CONFIG)
    
    # Visualize results
    plot_alpha_sensitivity(results, CONFIG)
    
    print("\nProgram completed successfully!")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()