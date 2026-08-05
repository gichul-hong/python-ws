import numpy as np
import pandas as pd
import os

# Load test probs
p1 = np.load('../iter10-gemini/test_probs.npy')  # ConvNeXt iter10
p2 = np.load('test_probs.npy')                   # Scratch CNN iter11

# Soft voting with weights (e.g. 0.65 for ConvNeXt, 0.35 for Scratch)
ensemble_p = 0.65 * p1 + 0.35 * p2
preds = np.argmax(ensemble_p, axis=1)

# Write result
df = pd.read_csv(r'C:\hong\python-ws\project-1\result.csv')
df['class'] = preds
df.to_csv('result_iter10_scratch.csv', index=False)
print("Ensemble completed: result_iter10_scratch.csv")

# Also print class distribution
counts = df['class'].value_counts().sort_index()
print(counts.to_dict())
