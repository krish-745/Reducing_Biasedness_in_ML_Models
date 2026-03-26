import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

def get_roc_curves(y_true, y_scores, groups):
    """Calculates ROC curves for each demographic group."""
    curves = {}
    unique_groups = np.unique(groups)
    for g in unique_groups:
        indices = (groups == g)
        fpr, tpr, thresholds = roc_curve(y_true[indices], y_scores[indices])
        curves[g] = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'threshold': thresholds})
    return curves

def find_fair_operating_point(curves):
    """
    Simplification: Finds a TPR/FPR point achievable by both groups.
    In the paper, this is done by finding the intersection of the 
    convex hulls of the ROC curves.
    """
    # For this example, we find a point where both groups have similar TPR
    # and then pick the max FPR between them to 'level down' to the fairer point.
    group_0 = curves[0]
    group_1 = curves[1]
    
    # Target a specific TPR (e.g., 0.7)
    target_tpr = 0.7
    
    # Find closest points in both curves to target_tpr
    p0 = group_0.iloc[(group_0['tpr'] - target_tpr).abs().argsort()[:1]]
    p1 = group_1.iloc[(group_1['tpr'] - target_tpr).abs().argsort()[:1]]
    
    return {
        0: {'threshold': p0['threshold'].values[0], 'fpr': p0['fpr'].values[0], 'tpr': p0['tpr'].values[0]},
        1: {'threshold': p1['threshold'].values[0], 'fpr': p1['fpr'].values[0], 'tpr': p1['tpr'].values[0]}
    }

# --- Example Usage ---

# 1. Synthetic Data
np.random.seed(42)
n = 1000
groups = np.random.randint(0, 2, n)
y_true = np.random.randint(0, 2, n)
# Group 1 has slightly higher scores (simulating historical bias)
y_scores = np.random.uniform(0, 1, n) + (groups * 0.1) + (y_true * 0.4)
y_scores = np.clip(y_scores, 0, 1)

# 2. Get Curves
curves = get_roc_curves(y_true, y_scores, groups)

# 3. Find Thresholds
fair_points = find_fair_operating_point(curves)

for g, metrics in fair_points.items():
    print(f"Group {g}: Use Threshold {metrics['threshold']:.3f} "
          f"-> (FPR: {metrics['fpr']:.2f}, TPR: {metrics['tpr']:.2f})")