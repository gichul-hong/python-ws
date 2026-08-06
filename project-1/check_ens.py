import pandas as pd
import numpy as np
oof_labels = np.load('iter11-gemini/oof_labels.npy')
tc = np.array([np.sum(oof_labels == i) for i in range(43)])
expected = (tc / len(oof_labels)) * 8670
for f in ['vote_all4_with_header.csv', 'vote_best3_with_header.csv', 'vote_weighted_with_header.csv']:
    df = pd.read_csv('ensemble-4people/' + f)
    pred_c = np.array([np.sum(df.iloc[:, -1] == i) for i in range(43)])
    diff = pred_c - expected
    print(f"{f}: sum|diff| = {np.sum(np.abs(diff)):.1f}, max|diff| = {np.max(np.abs(diff)):.1f}")
