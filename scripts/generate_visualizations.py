"""
Script to generate visualization placeholders for the interim report.

This script creates sample visualizations that match the report structure.
Run this script after processing your data to generate the visualization files.

Usage:
    # Activate virtual environment first
    source venv/bin/activate  # On Windows: venv\\Scripts\\activate
    python scripts/generate_visualizations.py
"""

import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
except ImportError as e:
    print("Error: Required packages not found. Please install dependencies:")
    print("  pip install matplotlib seaborn numpy pandas")
    print("\nOr activate your virtual environment:")
    print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
output_dir = Path("data/processed/eda_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating visualizations in {output_dir}...")

# 1. Class Distribution Comparison
print("1. Generating class distribution plot...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# E-commerce data (example)
ecommerce_classes = ['Normal', 'Fraud']
ecommerce_counts = [136961, 14151]
axes[0].bar(ecommerce_classes, ecommerce_counts, color=['skyblue', 'salmon'])
axes[0].set_title('E-commerce Dataset\nClass Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_xlabel('Class', fontsize=12)
for i, (cls, count) in enumerate(zip(ecommerce_classes, ecommerce_counts)):
    axes[0].text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')

# Banking data (example)
banking_classes = ['Normal', 'Fraud']
banking_counts = [283253, 473]
axes[1].bar(banking_classes, banking_counts, color=['skyblue', 'salmon'])
axes[1].set_title('Banking Dataset\nClass Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_xlabel('Class', fontsize=12)
for i, (cls, count) in enumerate(zip(banking_classes, banking_counts)):
    axes[1].text(i, count, f'{count:,}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig(output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'class_distribution.png'}")

# 2. Univariate Distributions
print("2. Generating univariate distributions...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

# Sample distributions
np.random.seed(42)
for i, (title, data) in enumerate([
    ('Purchase Value Distribution', np.random.normal(36.93, 18.32, 1000)),
    ('Time Since Signup (hours)', np.random.lognormal(7, 1, 1000)),
    ('Transaction Amount', np.random.exponential(88.29, 1000)),
    ('Age Distribution', np.random.normal(33, 8.62, 1000))
]):
    if i < len(axes):
        axes[i].hist(data, bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(title, fontsize=12, fontweight='bold')
        axes[i].set_xlabel(title.split()[0], fontsize=10)
        axes[i].set_ylabel('Frequency', fontsize=10)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'univariate_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'univariate_distributions.png'}")

# 3. Bivariate Correlations
print("3. Generating correlation heatmap...")
# Create sample correlation matrix
features = ['time_since_signup', 'txn_freq_1H', 'txn_freq_24H', 
            'purchase_value', 'age', 'country_risk', 'hour_of_day']
np.random.seed(42)
corr_matrix = np.random.rand(len(features), len(features))
corr_matrix = (corr_matrix + corr_matrix.T) / 2  # Make symmetric
np.fill_diagonal(corr_matrix, 1.0)
# Set time_since_signup to have strong negative correlation
corr_matrix[0, :] = -0.5
corr_matrix[:, 0] = -0.5
corr_matrix[0, 0] = 1.0

corr_df = pd.DataFrame(corr_matrix, index=features, columns=features)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_df, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=0.5, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Heatmap', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'bivariate_correlations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'bivariate_correlations.png'}")

# 4. Bivariate Boxplots
print("4. Generating bivariate boxplots...")
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Sample data for boxplots
np.random.seed(42)
for i, feature in enumerate(['time_since_signup', 'txn_freq_1H', 'txn_freq_24H', 
                             'purchase_value', 'age', 'hour_of_day']):
    if i < len(axes):
        normal_data = np.random.normal(50, 20, 100)
        fraud_data = np.random.normal(10, 5, 100) if feature == 'time_since_signup' else np.random.normal(60, 25, 100)
        
        data_to_plot = [normal_data, fraud_data]
        bp = axes[i].boxplot(data_to_plot, tick_labels=['Normal', 'Fraud'], patch_artist=True)
        bp['boxes'][0].set_facecolor('skyblue')
        bp['boxes'][1].set_facecolor('salmon')
        axes[i].set_title(f'{feature} by Class', fontsize=12, fontweight='bold')
        axes[i].set_ylabel(feature, fontsize=10)
        axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'bivariate_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'bivariate_boxplots.png'}")

print(f"\n✓ All visualizations generated successfully in {output_dir}")
print("\nNote: These are sample visualizations. For actual data visualizations,")
print("run the preprocessing pipeline with perform_eda=True")

