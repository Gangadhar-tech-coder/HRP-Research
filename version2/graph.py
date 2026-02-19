import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import os
from datetime import datetime

# ============================================
# INPUT YOUR RESULTS MANUALL
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
                'Dimensionality Ratio': k / features
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
# GRAPH 1: MAIN PREDICTION GRAPH
# Features on X-axis, Reduction % as different lines
# ============================================
print("\nCreating main prediction graph...")

fig, ax = plt.subplots(figsize=(16, 10))

# Plot line for each reduction percentage
colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']  # Red, Blue, Green, Orange
markers = ['o', 's', '^', 'D']
reduction_percentages = sorted(df['Reduction %'].unique())

for idx, reduction in enumerate(reduction_percentages):
    subset = df[df['Reduction %'] == reduction].sort_values('Features')
    
    ax.plot(subset['Features'], subset['Best Alpha'], 
            marker=markers[idx], 
            color=colors[idx],
            linewidth=3, 
            markersize=12,
            label=f'{reduction}% Reduction',
            markeredgecolor='black',
            markeredgewidth=2)
    
    # Add value labels
    for _, row in subset.iterrows():
        ax.annotate(f"{row['Best Alpha']:.1f}", 
                   (row['Features'], row['Best Alpha']),
                   xytext=(0, 10), textcoords='offset points',
                   ha='center', fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.4', facecolor=colors[idx], alpha=0.3))

# Add example prediction
ax.axvline(x=500, color='purple', linestyle='--', linewidth=2, alpha=0.5, label='Example: 500 features')
ax.text(500, 1.05, '500 features', ha='center', fontsize=12, fontweight='bold', color='purple')

ax.set_xlabel('Number of Features (Original Dimension)', fontsize=16, fontweight='bold')
ax.set_ylabel('Best Alpha Value', fontsize=16, fontweight='bold')
ax.set_title('🎯 ALPHA PREDICTION GRAPH 🎯\nFind Your Features → Follow the Line for Your Reduction % → Read Alpha',
             fontsize=18, fontweight='bold', pad=20)
ax.set_ylim(-0.1, 1.2)
ax.legend(fontsize=14, loc='best', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')

# Add usage instruction box
usage_text = """HOW TO USE:
1. Find your feature count on X-axis
2. Choose your reduction % line
3. Read alpha value on Y-axis
   
Example: 500 features, 50% reduction
→ Follow purple line to blue curve
→ Read alpha ≈ 0.X"""

ax.text(0.02, 0.98, usage_text, transform=ax.transAxes,
       fontsize=11, verticalalignment='top',
       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig(f'{output_folder}/main_prediction_graph.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: main_prediction_graph.png")

# ============================================
# GRAPH 2: HEATMAP VERSION
# ============================================
print("Creating heatmap version...")

# Create pivot table
pivot = df.pivot_table(values='Best Alpha', 
                       index='Features', 
                       columns='Reduction %', 
                       aggfunc='mean')

fig, ax = plt.subplots(figsize=(12, 10))

# Create heatmap
im = ax.imshow(pivot.values, cmap='RdYlGn_r', aspect='auto', 
               interpolation='bilinear', vmin=0, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(pivot.columns)))
ax.set_yticks(np.arange(len(pivot.index)))
ax.set_xticklabels([f'{int(x)}%' for x in pivot.columns], fontsize=14, fontweight='bold')
ax.set_yticklabels([f'{int(x)}' for x in pivot.index], fontsize=14, fontweight='bold')

# Add values to cells
for i in range(len(pivot.index)):
    for j in range(len(pivot.columns)):
        value = pivot.values[i, j]
        if not np.isnan(value):
            text_color = 'white' if value < 0.5 else 'black'
            ax.text(j, i, f'{value:.1f}',
                   ha="center", va="center",
                   color=text_color, fontsize=18, fontweight='bold')

# Highlight 500 features row if exists
if 500 in pivot.index.values:
    row_idx = list(pivot.index).index(500)
    ax.add_patch(plt.Rectangle((-0.5, row_idx-0.5), len(pivot.columns), 1,
                               fill=False, edgecolor='blue', linewidth=4))
    ax.text(len(pivot.columns), row_idx, '← 500 features', 
           fontsize=12, va='center', color='blue', fontweight='bold')

ax.set_xlabel('Dimensionality Reduction %', fontsize=16, fontweight='bold')
ax.set_ylabel('Number of Features', fontsize=16, fontweight='bold')
ax.set_title('📊 ALPHA LOOKUP TABLE 📊\nFind Row (Features) × Column (Reduction %) → Read Alpha',
             fontsize=18, fontweight='bold', pad=20)

# Add colorbar
cbar = plt.colorbar(im, ax=ax)
cbar.set_label('Best Alpha Value', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{output_folder}/heatmap_lookup_table.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: heatmap_lookup_table.png")

# ============================================
# GRAPH 3: 3D VISUALIZATION
# ============================================
print("Creating 3D visualization...")

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(111, projection='3d')

# Plot data points
scatter = ax.scatter(df['Features'], df['Reduction %'], df['Best Alpha'],
                    c=df['Best Alpha'], cmap='RdYlGn_r', s=300,
                    edgecolors='black', linewidth=2, alpha=0.8)

# Connect points with lines for each reduction %
for reduction in reduction_percentages:
    subset = df[df['Reduction %'] == reduction].sort_values('Features')
    ax.plot(subset['Features'], subset['Reduction %'], subset['Best Alpha'],
           linewidth=2, alpha=0.6)

# Add example point
if 500 in df['Features'].values and 50 in df['Reduction %'].values:
    example = df[(df['Features'] == 500) & (df['Reduction %'] == 50)]
    if not example.empty:
        ax.scatter([500], [50], [example['Best Alpha'].values[0]],
                  c='blue', s=500, marker='*', edgecolors='black', linewidth=3,
                  label=f'Example: 500 features, 50% reduction')

ax.set_xlabel('Features', fontsize=14, fontweight='bold', labelpad=10)
ax.set_ylabel('Reduction %', fontsize=14, fontweight='bold', labelpad=10)
ax.set_zlabel('Best Alpha', fontsize=14, fontweight='bold', labelpad=10)
ax.set_title('3D View: Features × Reduction % × Alpha', 
             fontsize=18, fontweight='bold', pad=20)

# Add colorbar
cbar = fig.colorbar(scatter, ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Alpha Value', fontsize=12, fontweight='bold')

plt.savefig(f'{output_folder}/3d_visualization.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: 3d_visualization.png")

# ============================================
# GRAPH 4: SEPARATE SUBPLOTS FOR EACH REDUCTION
# ============================================
print("Creating individual reduction plots...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

for idx, reduction in enumerate(reduction_percentages):
    ax = axes[idx]
    subset = df[df['Reduction %'] == reduction].sort_values('Features')
    
    # Bar plot
    bars = ax.bar(range(len(subset)), subset['Best Alpha'], 
                  color=colors[idx], edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels on bars
    for i, (_, row) in enumerate(subset.iterrows()):
        ax.text(i, row['Best Alpha'] + 0.05, f"{row['Best Alpha']:.1f}",
               ha='center', fontsize=12, fontweight='bold')
    
    # Set x-axis labels
    ax.set_xticks(range(len(subset)))
    ax.set_xticklabels([f"{int(row['Features'])}" for _, row in subset.iterrows()],
                       fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Best Alpha', fontsize=14, fontweight='bold')
    ax.set_xlabel('Number of Features', fontsize=14, fontweight='bold')
    ax.set_title(f'Reduction: {reduction}%', fontsize=16, fontweight='bold')
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Highlight 500 features if exists
    if 500 in subset['Features'].values:
        idx_500 = list(subset['Features']).index(500)
        bars[idx_500].set_edgecolor('blue')
        bars[idx_500].set_linewidth(4)

plt.tight_layout()
plt.savefig(f'{output_folder}/individual_reduction_plots.png', dpi=300, bbox_inches='tight')
print(f"✓ Saved: individual_reduction_plots.png")

# ============================================
# SAVE DATA TABLE
# ============================================
print("\nSaving data table...")

# Create summary table
summary = df.pivot_table(values='Best Alpha', 
                         index=['Features', 'Samples'], 
                         columns='Reduction %')
summary.to_excel(f'{output_folder}/alpha_values_table.xlsx')
print(f"✓ Saved: alpha_values_table.xlsx")

# ============================================
# PRINT PREDICTION EXAMPLE
# ============================================
print(f"\n{'='*80}")
print("EXAMPLE: HOW TO PREDICT ALPHA FOR 500x700 DATASET → 250 FEATURES")
print(f"{'='*80}\n")

# Find closest feature count
features_list = sorted(df['Features'].unique())
closest_features = min(features_list, key=lambda x: abs(x - 500))

print(f"Step 1: Your dataset has 500 features")
print(f"        Closest in data: {closest_features} features")

print(f"\nStep 2: You want to reduce to 250 features")
print(f"        Reduction = (250/500) × 100 = 50%")

print(f"\nStep 3: Look at the graph for:")
print(f"        X-axis = {closest_features} features")
print(f"        Line = 50% Reduction (Blue line)")

# Find the actual value if it exists
result = df[(df['Features'] == closest_features) & (df['Reduction %'] == 50)]
if not result.empty:
    alpha_value = result['Best Alpha'].values[0]
    print(f"\n🎯 PREDICTED ALPHA: {alpha_value:.1f}")
else:
    # Interpolate
    reduction_50 = df[df['Reduction %'] == 50].sort_values('Features')
    if len(reduction_50) >= 2:
        # Linear interpolation
        lower = reduction_50[reduction_50['Features'] <= 500].tail(1)
        upper = reduction_50[reduction_50['Features'] >= 500].head(1)
        
        if not lower.empty and not upper.empty:
            f1, a1 = lower['Features'].values[0], lower['Best Alpha'].values[0]
            f2, a2 = upper['Features'].values[0], upper['Best Alpha'].values[0]
            
            # Linear interpolation
            alpha_predicted = a1 + (a2 - a1) * (500 - f1) / (f2 - f1)
            print(f"\n🎯 PREDICTED ALPHA (interpolated): {alpha_predicted:.2f}")
            print(f"   Between {f1} features (α={a1:.1f}) and {f2} features (α={a2:.1f})")

print(f"\n{'='*80}")
print("ALL GRAPHS SAVED!")
print(f"{'='*80}\n")
print(f"Output folder: {output_folder}/\n")
print("Generated graphs:")
print("  1. main_prediction_graph.png ⭐ MAIN GRAPH - Use this!")
print("  2. heatmap_lookup_table.png - Quick lookup table")
print("  3. 3d_visualization.png - 3D perspective")
print("  4. individual_reduction_plots.png - Bar charts per reduction")
print("  5. alpha_values_table.xlsx - Data in Excel format")
print(f"\n{'='*80}\n")