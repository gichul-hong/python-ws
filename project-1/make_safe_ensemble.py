import numpy as np
import pandas as pd

# Load robust 3-fold models (No OOF tuning tricks)
conv_test = np.load('iter13-recovery/conv_test_probs.npy')
scratch_test = np.load('iter11-gemini/test_probs.npy')

# Safe, logic-based weights (Not OOF optimized)
safe_w = 0.65
safe_probs = safe_w * conv_test + (1 - safe_w) * scratch_test
safe_preds = np.argmax(safe_probs, axis=1)

# Generate safe CSV
df = pd.read_csv('result.csv')
df.iloc[:, -1] = safe_preds
df.to_csv('DS2_challenge_team1_final_SAFE.csv', index=False)

# Quick proxy check
oof_labels = np.load('iter11-gemini/oof_labels.npy')
tc = np.array([np.sum(oof_labels == i) for i in range(43)])
expected = (tc / len(oof_labels)) * 8670
pred_c = np.array([np.sum(safe_preds == i) for i in range(43)])
diff = pred_c - expected
print('Safe Ensemble sum|diff|:', np.sum(np.abs(diff)))
print('Safe Ensemble max|diff|:', np.max(np.abs(diff)))
print('Saved as DS2_challenge_team1_final_SAFE.csv')
