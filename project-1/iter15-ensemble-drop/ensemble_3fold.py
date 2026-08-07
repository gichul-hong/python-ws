import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import os

# 1. Load OOF (Out-of-Fold) probabilities and labels
models = ['conv', 'scratch', 'efficientnet', 'resnet', 'mobilenet']
oof_probs = []
test_probs = []

labels = np.load('conv_oof_labels.npy')

for m in models:
    oof_probs.append(np.load(f'{m}_oof_probs.npy'))
    test_probs.append(np.load(f'{m}_test_probs.npy'))

# 2. Find optimal global weight using OOF optimization
def objective(w):
    # w is a vector of 5 weights, we normalize it to sum to 1
    w = np.array(w)
    w = w / np.sum(w)
    
    ensemble_oof = np.zeros_like(oof_probs[0])
    for i in range(len(models)):
        ensemble_oof += w[i] * oof_probs[i]
        
    preds = np.argmax(ensemble_oof, axis=1)
    return -f1_score(labels, preds, average='macro')

print("Optimizing 5-model ensemble weights based on 3-Fold OOF...")
# Initial weights: equal
init_w = [0.2] * 5
bounds = [(0.0, 1.0)] * 5

res = minimize(objective, x0=init_w, bounds=bounds, method='Nelder-Mead')
opt_w = res.x / np.sum(res.x)

print("\n--- Optimal Weights ---")
for i, m in enumerate(models):
    print(f"{m:15s}: {opt_w[i]:.4f}")
print(f"Optimized OOF Macro F1: {-res.fun:.6f}")

# 4. Apply optimal weights to Test
final_test_probs = np.zeros_like(test_probs[0])
for i in range(len(models)):
    final_test_probs += opt_w[i] * test_probs[i]
preds = np.argmax(final_test_probs, axis=1)

# 5. Save Final CSV
df = pd.read_csv(r'C:\hong\python-ws\project-1\result.csv')
df['class'] = preds
df.to_csv('DS2_challenge_team1_final.csv', index=False)
print("\nFinal Optimized 5-Model Ensemble completed: DS2_challenge_team1_final.csv")

# Print class distribution
counts = df['class'].value_counts().sort_index()
print("\n--- Final Test Class Distribution ---")
print(counts.to_dict())
