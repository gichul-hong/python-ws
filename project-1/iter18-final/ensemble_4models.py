import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from scipy.optimize import minimize
import os

models = ['conv', 'resnet', 'efficientnet', 'mobilenet']
oof_list = []
test_list = []
for m in models:
    oof_list.append(np.load(f'{m}_oof_probs.npy'))
    test_list.append(np.load(f'{m}_test_probs.npy'))

labels = np.load('conv_oof_labels.npy')

def objective(w):
    w = np.array(w)
    w = w / np.sum(w)
    ensemble_oof = np.zeros_like(oof_list[0])
    for i, oof in enumerate(oof_list):
        ensemble_oof += w[i] * oof
    preds = np.argmax(ensemble_oof, axis=1)
    return -f1_score(labels, preds, average='macro')

print("Optimizing ensemble weights based on 5-Fold OOF...")
res = minimize(objective, x0=[0.25]*4, bounds=[(0.0, 1.0)]*4, method='Nelder-Mead')
opt_w = res.x / np.sum(res.x)

print(f"Optimal Weights: {dict(zip(models, np.round(opt_w, 4)))}")
print(f"Optimized OOF Macro F1:  {-res.fun:.6f}")

final_test_probs = np.zeros_like(test_list[0])
for i, test in enumerate(test_list):
    final_test_probs += opt_w[i] * test
preds = np.argmax(final_test_probs, axis=1)

df = pd.read_csv(r'C:\hong\python-ws\project-1\result.csv')
df['class'] = preds
df.to_csv('DS2_challenge_team1_final.csv', index=False)
print("\nFinal Optimized 4-Model Ensemble completed: DS2_challenge_team1_final.csv")
