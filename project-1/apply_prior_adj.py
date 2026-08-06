import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

# Load robust 3-fold models (No OOF tuning tricks)
conv_test = np.load('iter13-recovery/conv_test_probs.npy')
scratch_test = np.load('iter11-gemini/test_probs.npy')
conv_oof = np.load('iter13-recovery/conv_oof_probs.npy')
scratch_oof = np.load('iter11-gemini/oof_probs.npy')

safe_w = 0.65
test_probs = safe_w * conv_test + (1 - safe_w) * scratch_test
oof_probs = safe_w * conv_oof + (1 - safe_w) * scratch_oof
oof_labels = np.load('iter11-gemini/oof_labels.npy')

tc = np.array([np.sum(oof_labels == i) for i in range(43)])
expected = (tc / len(oof_labels)) * 8670
prior = tc / np.sum(tc)

for tau in [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]:
    # Adjust test probs
    adj_test = test_probs * (prior ** tau)
    adj_test /= adj_test.sum(axis=1, keepdims=True)
    preds_test = np.argmax(adj_test, axis=1)
    
    # Adjust OOF probs
    adj_oof = oof_probs * (prior ** tau)
    adj_oof /= adj_oof.sum(axis=1, keepdims=True)
    preds_oof = np.argmax(adj_oof, axis=1)
    
    oof_f1 = f1_score(oof_labels, preds_oof, average='macro')
    
    pred_c = np.array([np.sum(preds_test == i) for i in range(43)])
    diff = pred_c - expected
    sum_diff = np.sum(np.abs(diff))
    max_diff = np.max(np.abs(diff))
    
    print(f'Tau: {tau:.1f} | OOF F1: {oof_f1:.4f} | Proxy sum|diff|: {sum_diff:.1f} | max|diff|: {max_diff:.1f}')

    if tau == 0.4:
        df = pd.read_csv('result.csv')
        new_df = pd.DataFrame({'id': df['id'], 'class': preds_test})
        new_df.to_csv('result_prior_t04.csv', index=False)
        
    if tau == 0.6:
        df = pd.read_csv('result.csv')
        new_df = pd.DataFrame({'id': df['id'], 'class': preds_test})
        new_df.to_csv('result_prior_t06.csv', index=False)
