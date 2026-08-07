import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import os

# 1. Load OOF (Out-of-Fold) probabilities and labels
conv_oof = np.load('conv_oof_probs.npy')
scratch_oof = np.load('scratch_oof_probs.npy')
labels = np.load('conv_oof_labels.npy')  # Both models share the same track-level split labels

# 2. Find optimal global weight (or class-wise weights) using OOF optimization
def objective(w):
    # w: weight for ConvNeXt, (1-w) for Scratch CNN
    ensemble_oof = w * conv_oof + (1 - w) * scratch_oof
    preds = np.argmax(ensemble_oof, axis=1)
    # Minimize negative Macro F1
    return -f1_score(labels, preds, average='macro')

print("Optimizing ensemble weights based on 5-Fold OOF...")
res = minimize(objective, x0=[0.65], bounds=[(0.0, 1.0)], method='Nelder-Mead')
opt_w = res.x[0] if hasattr(res.x, '__len__') else res.x
print(f"Optimal ConvNeXt Weight: {opt_w:.4f} (Scratch: {1-opt_w:.4f})")
print(f"Optimized OOF Macro F1:  {-res.fun:.6f}")

# 3. Load Test probabilities
conv_test = np.load('conv_test_probs.npy')
scratch_test = np.load('scratch_test_probs.npy')

# 4. Apply optimal weights to Test
final_test_probs = opt_w * conv_test + (1 - opt_w) * scratch_test
preds = np.argmax(final_test_probs, axis=1)

# 5. Save Final CSV
df = pd.read_csv(r'C:\hong\python-ws\project-1\result.csv')
df['class'] = preds
df.to_csv('DS2_challenge_team1_final.csv', index=False)
print("\nFinal Optimized 5-Fold Ensemble completed: DS2_challenge_team1_final.csv")

# Print class distribution
counts = df['class'].value_counts().sort_index()
print("\n--- Final Test Class Distribution ---")
print(counts.to_dict())
