import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================
# INPUT YOUR RESULTS MANUALLY
# ============================================
print("="*80)
print("ALPHA PREDICTION GRAPH GENERATOR")
print("="*80)
print("\nEnter your experimental results from all datasets\n")

# Store all results
all_results = []

# Ask how many datasets
num_datasets = int(input("How many datasets do you have results for? ").strip())

for i in range(num_datasets):
    print(f"\n--- Dataset {i+1} ---")
    features = int(input("Number of features: ").strip())
    samples = int(input("Number of samples: ").strip())
    
    print(f"\nFor dataset {features}x{samples}, enter best alpha for each reduction:")
    
    # Get results for each reduction percentage
    for reduction_pct in [70, 50, 30, 10]:
        k = int(features * reduction_pct / 100)
        
        best_alpha = input(f"  Best alpha for {reduction_pct}% reduction (k={k}): ").strip()
        
        if best_alpha:  # If user entered a value
            all_results.append({
                'Dataset': f"{features}x{samples}",
                'Features': features,
                'Samples': samples,
                'Reduction %': reduction_pct,
                'k': k,
                'Best Alpha': float(best_alpha),
                'Dimensionality Ratio': k / features,
                'Sample to Feature Ratio': samples / features
            })

# Convert to DataFrame
df = pd.DataFrame(all_results)

print(f"\n✓ Collected {len(df)} data points from {num_datasets} datasets")

# ============================================
# CREATE OUTPUT FOLDER
# ============================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_folder = f"alpha_graphs_{timestamp}"
os.makedirs(output_folder, exist_ok=True)

# ============================================
# GRAPH 1: 3D SURFACE - Features × Samples × Alpha
# ============================================
print("\nCreating 3D prediction surface for each reduction %...")

reduction_percentages = sorted(df['Reduction %'].unique())

fig = plt.figure(figsize=(20, 16))

for idx, reduction in enumerate(reduction_percentages):
    ax = fig.add_subplot(2, 2, idx+1, projection='3d')
    
    subset = df[df['Reduction %'] == reduction]
    
    if len(subset) < 3:
        ax.text(0.5, 0.5, 0.5, 'Insufficient Data', ha='center', va='center', fontsize=14)
        ax.set_title(f'Reduction: {reduction}%', fontsize=14, fontweight='bold')
        continue
    
    # Create grid for interpolation
    features_range = np.linspace(df['Features'].min(), df['Features'].max(), 50)
    samples_range = np.linspace(df['Samples'].min(), df['Samples'].max(), 50)
    features_grid, samples_grid = np.meshgrid(features_range, samples_range)
    
    # Interpolate alpha values
    try:
        alpha_grid = griddata(
            (subset['Features'], subset['Samples']),
            subset['Best Alpha'],
            (features_grid, samples_grid),
            method='cubic',
            fill_value=subset['Best Alpha'].mean()
        )
    except:
        alpha_grid = griddata(
            (subset['Features'], subset['Samples']),
            subset['Best Alpha'],
            (features_grid, samples_grid),
            method='linear',
            fill_value=subset['Best Alpha'].mean()
        )
    
    # Plot surface
    surf = ax.plot_surface(features_grid, samples_grid, alpha_grid,
                           cmap='RdYlGn_r', alpha=0.6, edgecolor='none')
    
    # Plot actual data points
    ax.scatter(subset['Features'], subset['Samples'], subset['Best Alpha'],
              c='red', s=200, edgecolors='black', linewidth=2, zorder=10)
    
    # Add labels to points
    for _, row in subset.iterrows():
        ax.text(row['Features'], row['Samples'], row['Best Alpha'],
               f"{row['Best Alpha']:.1f}",
               fontsize=9, color='black')
    
    ax.set_xlabel('Features', fontsize=12, labelpad=8)
    ax.set_ylabel('Samples', fontsize=12, labelpad=8)
    ax.set_zlabel('Best Alpha', fontsize=12, labelpad=8)
    ax.set_title(f'Reduction: {reduction}%', fontsize=14, fontweight='bold')
    ax.set_zlim(0, 1)

plt.suptitle('3D Prediction Surfaces: Features × Samples → Best Alpha',
             fontsize=18, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig(f'{output_folder}/1_3d_surfaces.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 1_3d_surfaces.png")
plt.close()

# ============================================
# GRAPH 2: CONTOUR MAPS (TOP VIEW) - MAIN PREDICTION TOOL
# ============================================
print("Creating contour prediction maps...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, reduction in enumerate(reduction_percentages):
    ax = axes[idx]
    
    subset = df[df['Reduction %'] == reduction]
    
    if len(subset) < 3:
        ax.text(0.5, 0.5, 'Insufficient Data\n(Need at least 3 points)',
               ha='center', va='center', transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Reduction: {reduction}%', fontsize=14, fontweight='bold')
        continue
    
    # Create grid
    features_range = np.linspace(df['Features'].min(), df['Features'].max(), 100)
    samples_range = np.linspace(df['Samples'].min(), df['Samples'].max(), 100)
    features_grid, samples_grid = np.meshgrid(features_range, samples_range)
    
    # Interpolate
    try:
        alpha_grid = griddata(
            (subset['Features'], subset['Samples']),
            subset['Best Alpha'],
            (features_grid, samples_grid),
            method='cubic',
            fill_value=subset['Best Alpha'].mean()
        )
    except:
        alpha_grid = griddata(
            (subset['Features'], subset['Samples']),
            subset['Best Alpha'],
            (features_grid, samples_grid),
            method='linear',
            fill_value=subset['Best Alpha'].mean()
        )
    
    # Create filled contour
    contour_filled = ax.contourf(features_grid, samples_grid, alpha_grid,
                                  levels=20, cmap='RdYlGn_r', alpha=0.8)
    
    # Add contour lines with labels (FIXED - removed fontweight)
    contour_lines = ax.contour(features_grid, samples_grid, alpha_grid,
                               levels=10, colors='black', linewidths=1.5, alpha=0.4)
    ax.clabel(contour_lines, inline=True, fontsize=10, fmt='%.1f')  # FIXED
    
    # Plot actual data points
    scatter = ax.scatter(subset['Features'], subset['Samples'],
                        c=subset['Best Alpha'], s=400, cmap='RdYlGn_r',
                        edgecolors='black', linewidth=3, zorder=10)
    
    # Add value labels
    for _, row in subset.iterrows():
        ax.annotate(f"{row['Best Alpha']:.1f}",
                   (row['Features'], row['Samples']),
                   xytext=(0, 0), textcoords='offset points',
                   fontsize=11, ha='center', va='center',
                   fontweight='bold', color='white',
                   bbox=dict(boxstyle='circle,pad=0.3', facecolor='black', alpha=0.7))
    
    ax.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_title(f'Reduction: {reduction}%\nFind (Features, Samples) → Read Alpha from Contours',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(contour_filled, ax=ax)
    cbar.set_label('Best Alpha', fontsize=12, fontweight='bold')

plt.suptitle('🎯 CONTOUR PREDICTION MAPS 🎯\nHow to Use: Find (Features, Samples) → Read Alpha Value',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_folder}/2_contour_maps_MAIN.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 2_contour_maps_MAIN.png ⭐ MAIN PREDICTION TOOL")
plt.close()

# ============================================
# GRAPH 3: HEATMAP GRID
# ============================================
print("Creating heatmap grids...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, reduction in enumerate(reduction_percentages):
    ax = axes[idx]
    
    subset = df[df['Reduction %'] == reduction]
    
    # Create pivot table
    pivot = subset.pivot_table(values='Best Alpha',
                               index='Samples',
                               columns='Features',
                               aggfunc='mean')
    
    if pivot.empty:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center',
               transform=ax.transAxes, fontsize=14)
        ax.set_title(f'Reduction: {reduction}%', fontsize=14, fontweight='bold')
        continue
    
    # Create heatmap
    sns.heatmap(pivot, annot=True, fmt='.1f', cmap='RdYlGn_r',
               ax=ax, cbar_kws={'label': 'Best Alpha'},
               linewidths=2, linecolor='black',
               vmin=0, vmax=1, annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_xlabel('Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Samples', fontsize=14, fontweight='bold')
    ax.set_title(f'Reduction: {reduction}%\nLookup Table Format',
                fontsize=14, fontweight='bold')
    
    # Rotate labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

plt.suptitle('📊 LOOKUP TABLES 📊\nFind Row (Samples) × Column (Features) → Read Alpha',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_folder}/3_heatmap_lookup.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 3_heatmap_lookup.png")
plt.close()

# ============================================
# GRAPH 4: FEATURE-GROUPED WITH SAMPLE VARIATIONS
# ============================================
print("Creating feature-grouped comparison...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

colors = plt.cm.tab10(np.linspace(0, 1, 10))

for idx, reduction in enumerate(reduction_percentages):
    ax = axes[idx]
    
    subset = df[df['Reduction %'] == reduction]
    
    # Group by features, plot samples on x-axis
    unique_features = sorted(subset['Features'].unique())
    
    for f_idx, features in enumerate(unique_features):
        feature_data = subset[subset['Features'] == features].sort_values('Samples')
        
        if len(feature_data) > 0:
            ax.plot(feature_data['Samples'], feature_data['Best Alpha'],
                   marker='o', linewidth=3, markersize=12,
                   label=f'{features} features',
                   color=colors[f_idx % len(colors)],
                   markeredgecolor='black', markeredgewidth=2)
            
            # Add value labels
            for _, row in feature_data.iterrows():
                ax.annotate(f"{row['Best Alpha']:.1f}",
                          (row['Samples'], row['Best Alpha']),
                          xytext=(0, 10), textcoords='offset points',
                          fontsize=9, ha='center',
                          bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor=colors[f_idx % len(colors)],
                                   alpha=0.3))
    
    ax.set_xlabel('Number of Samples', fontsize=14, fontweight='bold')
    ax.set_ylabel('Best Alpha', fontsize=14, fontweight='bold')
    ax.set_title(f'Reduction: {reduction}%\nHow Sample Count Affects Alpha',
                fontsize=14, fontweight='bold')
    ax.set_ylim(-0.1, 1.1)
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3)

plt.suptitle('Sample Count Effect on Alpha\nSeparate Lines for Each Feature Count',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_folder}/4_sample_variation.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 4_sample_variation.png")
plt.close()

# ============================================
# GRAPH 5: BUBBLE CHART
# ============================================
print("Creating bubble chart overview...")

fig, axes = plt.subplots(2, 2, figsize=(20, 16))
axes = axes.flatten()

for idx, reduction in enumerate(reduction_percentages):
    ax = axes[idx]
    
    subset = df[df['Reduction %'] == reduction]
    
    # Bubble size based on alpha value
    sizes = (subset['Best Alpha'] * 1000) + 100
    
    scatter = ax.scatter(subset['Features'], subset['Samples'],
                        s=sizes, c=subset['Best Alpha'],
                        cmap='RdYlGn_r', alpha=0.6,
                        edgecolors='black', linewidth=2)
    
    # Add labels
    for _, row in subset.iterrows():
        ax.annotate(f"{row['Dataset']}\nα={row['Best Alpha']:.1f}",
                   (row['Features'], row['Samples']),
                   fontsize=9, ha='center', va='center',
                   fontweight='bold')
    
    ax.set_xlabel('Features', fontsize=14, fontweight='bold')
    ax.set_ylabel('Samples', fontsize=14, fontweight='bold')
    ax.set_title(f'Reduction: {reduction}%\nBubble Size = Alpha Value',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Use log scale only if range is large
    if subset['Features'].max() / subset['Features'].min() > 10:
        ax.set_xscale('log')
    if subset['Samples'].max() / subset['Samples'].min() > 10:
        ax.set_yscale('log')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Best Alpha', fontsize=12, fontweight='bold')

plt.suptitle('Bubble Chart Overview: Dataset Characteristics vs Alpha',
             fontsize=18, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig(f'{output_folder}/5_bubble_overview.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 5_bubble_overview.png")
plt.close()

# ============================================
# SAVE DATA TABLE
# ============================================
print("\nSaving data table...")
df.to_excel(f'{output_folder}/alpha_values_complete.xlsx', index=False)
print(f"✓ Saved: alpha_values_complete.xlsx")

# ============================================
# PREDICTION EXAMPLE
# ============================================
print(f"\n{'='*80}")
print("PREDICTION EXAMPLES")
print(f"{'='*80}\n")

# Example predictions
test_cases = [
    (10, 15000, "10 features, 15000 samples"),
    (50, 5000, "50 features, 5000 samples"),
    (100, 10000, "100 features, 10000 samples")
]

for features_test, samples_test, description in test_cases:
    print(f"\n{description}:")
    print("-" * 60)
    
    for reduction in reduction_percentages:
        subset = df[df['Reduction %'] == reduction]
        
        if len(subset) < 3:
            print(f"  Reduction {reduction}%: Insufficient data")
            continue
        
        # Try cubic interpolation first
        try:
            predicted_alpha = griddata(
                (subset['Features'], subset['Samples']),
                subset['Best Alpha'],
                (features_test, samples_test),
                method='cubic'
            )
        except:
            predicted_alpha = np.nan
        
        # If cubic fails, try linear
        if np.isnan(predicted_alpha):
            try:
                predicted_alpha = griddata(
                    (subset['Features'], subset['Samples']),
                    subset['Best Alpha'],
                    (features_test, samples_test),
                    method='linear'
                )
            except:
                predicted_alpha = np.nan
        
        # If linear fails, try nearest
        if np.isnan(predicted_alpha):
            try:
                predicted_alpha = griddata(
                    (subset['Features'], subset['Samples']),
                    subset['Best Alpha'],
                    (features_test, samples_test),
                    method='nearest'
                )
            except:
                predicted_alpha = np.nan
        
        if not np.isnan(predicted_alpha):
            print(f"  Reduction {reduction}%: Predicted α = {predicted_alpha:.2f}")
        else:
            print(f"  Reduction {reduction}%: Cannot interpolate (outside data range)")

print(f"\n{'='*80}")
print("HOW TO USE THE CONTOUR MAP (2_contour_maps_MAIN.png):")
print(f"{'='*80}")
print("""
1. Choose your reduction % subplot (70%, 50%, 30%, or 10%)
2. Find your feature count on X-axis (e.g., 10)
3. Find your sample count on Y-axis (e.g., 15000)
4. Look at the intersection point
5. Read the alpha value from:
   - The contour line labels (black numbers)
   - The color (check colorbar on right)
   - Nearby data points (white numbers in black circles)

Example: 10 features × 15000 samples, 50% reduction
→ Go to subplot "Reduction: 50%"
→ Find x=10, y=15000
→ Read alpha at that point
""")

print(f"\n{'='*80}")
print("ALL GRAPHS SAVED!")
print(f"{'='*80}\n")
print(f"Output folder: {output_folder}/\n")
print("Generated graphs:")
print("  1. 1_3d_surfaces.png - 3D view of prediction surface")
print("  2. 2_contour_maps_MAIN.png ⭐ MAIN TOOL - Contour maps for prediction")
print("  3. 3_heatmap_lookup.png - Quick lookup tables")
print("  4. 4_sample_variation.png - How samples affect alpha")
print("  5. 5_bubble_overview.png - Overall pattern visualization")
print("  6. alpha_values_complete.xlsx - All data in Excel")
print(f"\n{'='*80}\n")