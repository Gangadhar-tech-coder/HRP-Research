"""
Hybrid Random Projection (HRP) – Alpha Sensitivity Study
Author: You

Goal:
- Vary alpha from 0.1 to 1.0
- Compute distance distortion for:
  1. Normal Random Projection (NRP)
  2. Plus-Minus One Random Projection (PMRP)
  3. Hybrid Random Projection (HRP)
- Visualize all results together

Requirements:
pip install numpy scipy matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist


# =========================================================
# 1. RANDOM PROJECTION MATRICES
# =========================================================

def normal_random_projection(d, k):
    return np.random.normal(0, 1, size=(d, k))


def plus_minus_one_projection(d, k):
    return np.random.choice([-1, 1], size=(d, k))


# =========================================================
# 2. DISTANCE DISTORTION (ERROR METRIC)
# =========================================================

def distance_distortion(X, Y):
    original_distances = pdist(X, metric="euclidean")
    projected_distances = pdist(Y, metric="euclidean")
    ratio = projected_distances / original_distances
    return np.mean(np.abs(ratio - 1))


# =========================================================
# 3. HYBRID PROJECTION
# =========================================================

def hybrid_projection(R1, R2, alpha):
    return alpha * R1 + (1 - alpha) * R2


# =========================================================
# 4. MAIN HRP EXPERIMENT (NRP + PMRP + HRP)
# =========================================================

def hrp_full_experiment(X, k, alpha_values):
    n, d = X.shape

    # Generate projection matrices ONCE (important!)
    R_nrp = normal_random_projection(d, k)
    R_pmrp = plus_minus_one_projection(d, k)

    nrp_distortions = []
    pmrp_distortions = []
    hrp_distortions = []

    for alpha in alpha_values:
        # ----- NRP -----
        Y_nrp = X @ R_nrp
        d_nrp = distance_distortion(X, Y_nrp)

        # ----- PMRP -----
        Y_pmrp = X @ R_pmrp
        d_pmrp = distance_distortion(X, Y_pmrp)

        # ----- HRP -----
        H = hybrid_projection(R_nrp, R_pmrp, alpha)
        Y_hrp = X @ H
        d_hrp = distance_distortion(X, Y_hrp)

        # Store
        nrp_distortions.append(d_nrp)
        pmrp_distortions.append(d_pmrp)
        hrp_distortions.append(d_hrp)

        # Display values
        print(
            f"α={alpha:.1f} | "
            f"NRP={d_nrp:.4f} | "
            f"PMRP={d_pmrp:.4f} | "
            f"HRP={d_hrp:.4f}"
        )

    return nrp_distortions, pmrp_distortions, hrp_distortions


# =========================================================
# 5. VISUALIZATION
# =========================================================

def plot_results(alpha_values, nrp, pmrp, hrp, n, d):
    plt.figure(figsize=(9, 6))

    plt.plot(alpha_values, nrp, marker='o', label="NRP")
    plt.plot(alpha_values, pmrp, marker='s', label="PMRP")
    plt.plot(alpha_values, hrp, marker='^', label="HRP")

    plt.xlabel("Alpha (α)")
    plt.ylabel("Distance Distortion")
    plt.title(f"Alpha vs Distortion (n={n}, d={d})")
    plt.legend()
    plt.grid(True)
    plt.show()


# =========================================================
# 6. RESEARCH DRIVER
# =========================================================

def run_hrp_research():
    np.random.seed(42)

    sample_sizes = [50]
    dimensions = [200]
    reduced_dim_ratio = 0.3

    # Alpha from 0.1 to 1.0
    alpha_values = np.arange(0.1, 1.01, 0.1)

    for n in sample_sizes:
        for d in dimensions:
            print(f"\n===== HRP Experiment (n={n}, d={d}) =====")

            X = np.random.rand(n, d)
            k = int(d * reduced_dim_ratio)

            nrp, pmrp, hrp = hrp_full_experiment(X, k, alpha_values)

            plot_results(alpha_values, nrp, pmrp, hrp, n, d)


# =========================================================
# 7. ENTRY POINT
# =========================================================

if __name__ == "__main__":
    run_hrp_research()
