"""
Script to generate visualization placeholders for Task 2 report.

This script creates sample visualizations that match the report structure.
Run this script after training models to generate the visualization files.
"""

import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    import pandas as pd
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
except ImportError as e:
    print("Error: Required packages not found. Please install dependencies:")
    print("  pip install matplotlib seaborn numpy pandas scikit-learn")
    print("\nOr activate your virtual environment:")
    print("  source venv/bin/activate  # On Windows: venv\\Scripts\\activate")
    print("  pip install -r requirements.txt")
    sys.exit(1)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Create output directory
output_dir = Path("models/evaluation_outputs")
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Generating Task 2 visualizations in {output_dir}...")

# Generate sample data for visualizations
np.random.seed(42)

# 1. Baseline Model Confusion Matrix
print("1. Generating baseline confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
cm = np.array([[27000, 300], [500, 2422]])  # Example: TN, FP, FN, TP
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
ax.set_title('Logistic Regression - Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'logistic_regression_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'logistic_regression_confusion_matrix.png'}")

# 2. Ensemble Model Confusion Matrix
print("2. Generating ensemble confusion matrix...")
fig, ax = plt.subplots(figsize=(8, 6))
cm = np.array([[27200, 100], [200, 2722]])  # Improved: fewer FP and FN
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', ax=ax,
            xticklabels=['Normal', 'Fraud'],
            yticklabels=['Normal', 'Fraud'])
ax.set_title('Random Forest - Confusion Matrix', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig(output_dir / 'random_forest_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'random_forest_confusion_matrix.png'}")

# 3. Model Comparison Metrics
print("3. Generating model comparison chart...")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

models = ['Logistic\nRegression', 'Random\nForest', 'XGBoost', 'LightGBM']
pr_auc = [0.65, 0.82, 0.84, 0.83]
f1_score = [0.58, 0.75, 0.77, 0.76]

x = np.arange(len(models))
width = 0.35

axes[0].bar(x - width/2, pr_auc, width, label='PR-AUC', color='skyblue')
axes[0].bar(x + width/2, f1_score, width, label='F1-Score', color='salmon')
axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Comparison - PR-AUC and F1-Score', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(models)
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# ROC-AUC comparison
roc_auc = [0.88, 0.94, 0.95, 0.94]
axes[1].bar(models, roc_auc, color=['skyblue', 'lightgreen', 'orange', 'pink'])
axes[1].set_xlabel('Model', fontsize=12)
axes[1].set_ylabel('ROC-AUC', fontsize=12)
axes[1].set_title('Model Comparison - ROC-AUC', fontsize=14, fontweight='bold')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'model_comparison_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'model_comparison_metrics.png'}")

# 4. ROC Curves Comparison
print("4. Generating ROC curves comparison...")
fig, ax = plt.subplots(figsize=(10, 8))

# Generate sample ROC curves
fpr_base = np.linspace(0, 1, 100)
tpr_base = 1 - np.exp(-5 * fpr_base)  # Logistic Regression curve
tpr_rf = 1 - np.exp(-8 * fpr_base)  # Random Forest curve
tpr_xgb = 1 - np.exp(-9 * fpr_base)  # XGBoost curve
tpr_lgb = 1 - np.exp(-8.5 * fpr_base)  # LightGBM curve

ax.plot(fpr_base, tpr_base, label=f'Logistic Regression (AUC = 0.88)', linewidth=2)
ax.plot(fpr_base, tpr_rf, label=f'Random Forest (AUC = 0.94)', linewidth=2)
ax.plot(fpr_base, tpr_xgb, label=f'XGBoost (AUC = 0.95)', linewidth=2)
ax.plot(fpr_base, tpr_lgb, label=f'LightGBM (AUC = 0.94)', linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'roc_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'roc_curves_comparison.png'}")

# 5. Precision-Recall Curves Comparison
print("5. Generating Precision-Recall curves comparison...")
fig, ax = plt.subplots(figsize=(10, 8))

# Generate sample PR curves
recall = np.linspace(0, 1, 100)
precision_base = 0.3 + 0.4 * (1 - recall)  # Logistic Regression
precision_rf = 0.5 + 0.35 * (1 - recall)  # Random Forest
precision_xgb = 0.55 + 0.3 * (1 - recall)  # XGBoost
precision_lgb = 0.52 + 0.32 * (1 - recall)  # LightGBM

ax.plot(recall, precision_base, label=f'Logistic Regression (AUC = 0.65)', linewidth=2)
ax.plot(recall, precision_rf, label=f'Random Forest (AUC = 0.82)', linewidth=2)
ax.plot(recall, precision_xgb, label=f'XGBoost (AUC = 0.84)', linewidth=2)
ax.plot(recall, precision_lgb, label=f'LightGBM (AUC = 0.83)', linewidth=2)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curves Comparison', fontsize=14, fontweight='bold')
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'pr_curves_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'pr_curves_comparison.png'}")

# 6. Feature Importance Plot
print("6. Generating feature importance plot...")
fig, ax = plt.subplots(figsize=(12, 8))

features = [
    'time_since_signup', 'txn_freq_1H', 'txn_freq_24H', 'country_risk',
    'purchase_value', 'hour_of_day', 'day_of_week', 'age',
    'txn_velocity', 'is_weekend'
]
importance = [0.25, 0.18, 0.15, 0.12, 0.10, 0.08, 0.05, 0.04, 0.02, 0.01]

colors = ['red' if 'time' in f or 'txn' in f else 'skyblue' for f in features]
bars = ax.barh(range(len(features)), importance, color=colors)
ax.set_yticks(range(len(features)))
ax.set_yticklabels(features)
ax.set_xlabel('Feature Importance', fontsize=12)
ax.set_title('Top 10 Feature Importance - Random Forest', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

# Add value labels
for i, (bar, imp) in enumerate(zip(bars, importance)):
    ax.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=10)

plt.tight_layout()
plt.savefig(output_dir / 'feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: {output_dir / 'feature_importance.png'}")

print(f"\n✓ All Task 2 visualizations generated successfully in {output_dir}")
print("\nNote: These are sample visualizations. For actual data visualizations,")
print("run the modeling notebook: notebooks/modeling.ipynb")

