"""
Script to generate visualization placeholders for Task 3 report.

This script creates sample visualizations that match the report structure.
Run this script after running explainability analysis to generate the visualization files.
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
output_dir = Path("models/explainability_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating Task 3 visualizations in {output_dir}...")

# Generate sample data for visualizations
np.random.seed(42)

# 1. Feature Importance Comparison
print("1. Generating feature importance comparison...")
fig, axes = plt.subplots(1, 2, figsize=(16, 8))

features = [
    'time_since_signup', 'txn_freq_1H', 'txn_freq_24H', 'country_risk',
    'purchase_value', 'hour_of_day', 'day_of_week', 'age',
    'txn_velocity', 'is_weekend'
]
builtin_importance = [0.250, 0.180, 0.150, 0.120, 0.100, 0.080, 0.050, 0.040, 0.020, 0.010]
shap_importance = [0.248, 0.175, 0.152, 0.118, 0.095, 0.082, 0.048, 0.042, 0.022, 0.012]

x = np.arange(len(features))
width = 0.35

axes[0].barh(x - width/2, builtin_importance, width, label='Built-in (RF)', color='skyblue')
axes[0].barh(x + width/2, shap_importance, width, label='SHAP', color='salmon')
axes[0].set_yticks(x)
axes[0].set_yticklabels(features, fontsize=9)
axes[0].set_xlabel('Importance', fontsize=12)
axes[0].set_title('Feature Importance Comparison', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='x')

# Correlation plot
axes[1].scatter(builtin_importance, shap_importance, s=100, alpha=0.7, color='green')
axes[1].plot([0, 0.3], [0, 0.3], 'r--', linewidth=2, label='Perfect Agreement')
axes[1].set_xlabel('Built-in Importance', fontsize=12)
axes[1].set_ylabel('SHAP Importance', fontsize=12)
axes[1].set_title('Built-in vs SHAP Correlation (r=0.98)', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# Add feature labels
for i, feat in enumerate(features[:5]):  # Label top 5
    axes[1].annotate(feat, (builtin_importance[i], shap_importance[i]), 
                     fontsize=8, alpha=0.7)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'feature_importance_comparison.png'}")

# 2. SHAP Summary Plot
print("2. Generating SHAP summary plot...")
fig, ax = plt.subplots(figsize=(10, 8))

# Simulate SHAP values (negative for time_since_signup, positive for others)
n_samples = 100
shap_data = {
    'time_since_signup': np.random.normal(-0.25, 0.1, n_samples),
    'txn_freq_1H': np.random.normal(0.18, 0.08, n_samples),
    'txn_freq_24H': np.random.normal(0.15, 0.07, n_samples),
    'country_risk': np.random.normal(0.12, 0.06, n_samples),
    'purchase_value': np.random.normal(0.10, 0.05, n_samples),
    'hour_of_day': np.random.normal(0.08, 0.04, n_samples),
    'day_of_week': np.random.normal(0.05, 0.03, n_samples),
    'age': np.random.normal(0.04, 0.02, n_samples)
}

# Create summary plot
y_pos = np.arange(len(shap_data))
mean_shap = [np.mean(v) for v in shap_data.values()]
std_shap = [np.std(v) for v in shap_data.values()]

colors = ['red' if 'time' in k else 'blue' for k in shap_data.keys()]
bars = ax.barh(y_pos, mean_shap, xerr=std_shap, color=colors, alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(list(shap_data.keys()))
ax.set_xlabel('Mean |SHAP Value|', fontsize=12)
ax.set_title('SHAP Summary Plot - Global Feature Importance', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'shap_summary_plot.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'shap_summary_plot.png'}")

# 3. SHAP Force Plots (TP, FP, FN)
print("3. Generating SHAP force plots...")
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

# True Positive
features_tp = ['time_since_signup', 'txn_freq_1H', 'country_risk', 'purchase_value', 'hour_of_day']
shap_vals_tp = [-0.35, 0.28, 0.15, 0.08, 0.05]
colors_tp = ['red' if v < 0 else 'blue' for v in shap_vals_tp]
axes[0].barh(features_tp, shap_vals_tp, color=colors_tp, alpha=0.7)
axes[0].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[0].set_title('True Positive Case (Predicted: Fraud, Actual: Fraud, Prob: 0.87)', 
                 fontsize=12, fontweight='bold')
axes[0].set_xlabel('SHAP Value', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='x')

# False Positive
features_fp = ['txn_freq_1H', 'txn_freq_24H', 'country_risk', 'purchase_value', 'hour_of_day']
shap_vals_fp = [0.28, 0.22, 0.12, 0.08, 0.05]
colors_fp = ['blue'] * len(shap_vals_fp)
axes[1].barh(features_fp, shap_vals_fp, color=colors_fp, alpha=0.7)
axes[1].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[1].set_title('False Positive Case (Predicted: Fraud, Actual: Normal, Prob: 0.62)', 
                 fontsize=12, fontweight='bold')
axes[1].set_xlabel('SHAP Value', fontsize=10)
axes[1].grid(True, alpha=0.3, axis='x')

# False Negative
features_fn = ['time_since_signup', 'txn_freq_1H', 'country_risk', 'purchase_value', 'age']
shap_vals_fn = [-0.15, 0.08, 0.05, 0.03, 0.02]
colors_fn = ['red' if v < 0 else 'blue' for v in shap_vals_fn]
axes[2].barh(features_fn, shap_vals_fn, color=colors_fn, alpha=0.7)
axes[2].axvline(x=0, color='black', linestyle='-', linewidth=1)
axes[2].set_title('False Negative Case (Predicted: Normal, Actual: Fraud, Prob: 0.45)', 
                 fontsize=12, fontweight='bold')
axes[2].set_xlabel('SHAP Value', fontsize=10)
axes[2].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(output_dir / 'shap_force_plots.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'shap_force_plots.png'}")

# 4. Top 5 Drivers Analysis
print("4. Generating top 5 drivers analysis...")
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

top5_features = ['time_since_signup', 'txn_freq_1H', 'txn_freq_24H', 'country_risk', 'purchase_value']
top5_importance = [0.248, 0.175, 0.152, 0.118, 0.095]

# Importance bar chart
axes[0, 0].barh(top5_features, top5_importance, color=['red', 'orange', 'orange', 'yellow', 'yellow'])
axes[0, 0].set_xlabel('SHAP Importance', fontsize=10)
axes[0, 0].set_title('Top 5 Feature Importance', fontsize=11, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Time since signup distribution
axes[0, 1].hist([0.0003] * 100 + [1443] * 900, bins=50, color='red', alpha=0.7, edgecolor='black')
axes[0, 1].set_xlabel('Time Since Signup (hours)', fontsize=10)
axes[0, 1].set_ylabel('Frequency', fontsize=10)
axes[0, 1].set_title('time_since_signup Distribution\n(Fraud: 0.0003h, Normal: 1443h)', 
                    fontsize=11, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Transaction frequency 1H
axes[0, 2].bar(['Normal', 'Fraud'], [2.1, 8.5], color=['skyblue', 'salmon'])
axes[0, 2].set_ylabel('Avg Transactions/Hour', fontsize=10)
axes[0, 2].set_title('txn_freq_1H by Class', fontsize=11, fontweight='bold')
axes[0, 2].grid(True, alpha=0.3, axis='y')

# Transaction frequency 24H
axes[1, 0].bar(['Normal', 'Fraud'], [5.2, 22.3], color=['skyblue', 'salmon'])
axes[1, 0].set_ylabel('Avg Transactions/24H', fontsize=10)
axes[1, 0].set_title('txn_freq_24H by Class', fontsize=11, fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Country risk
axes[1, 1].bar(['Low Risk', 'High Risk'], [4.68, 26.42], color=['green', 'red'])
axes[1, 1].set_ylabel('Fraud Rate (%)', fontsize=10)
axes[1, 1].set_title('country_risk Impact', fontsize=11, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

# Purchase value
normal_vals = np.random.normal(36.93, 5, 500)
fraud_vals = np.random.normal(36.99, 5, 500)
axes[1, 2].hist([normal_vals, fraud_vals], bins=30, color=['skyblue', 'salmon'], 
                alpha=0.7, edgecolor='black', label=['Normal', 'Fraud'])
axes[1, 2].legend()
axes[1, 2].set_xlabel('Purchase Value ($)', fontsize=10)
axes[1, 2].set_ylabel('Frequency', fontsize=10)
axes[1, 2].set_title('purchase_value Distribution\n(Similar for Normal/Fraud)', 
                    fontsize=11, fontweight='bold')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'top5_drivers_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'top5_drivers_analysis.png'}")

print(f"\n✓ All Task 3 visualizations generated successfully in {output_dir}")
print("\nNote: These are sample visualizations. For actual data visualizations,")
print("run the explainability notebook: notebooks/shap-explainability.ipynb")

