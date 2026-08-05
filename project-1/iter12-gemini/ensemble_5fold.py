import numpy as np
import pandas as pd
import os

# Load test probs from 5-fold models
p1 = np.load('conv_test_probs.npy')      # ConvNeXt 5-fold (128px)
p2 = np.load('scratch_test_probs.npy')   # Scratch CNN 5-fold (48px)

# Soft voting (ConvNeXt is stronger, so weight 0.65 vs 0.35)
ensemble_p = 0.65 * p1 + 0.35 * p2
preds = np.argmax(ensemble_p, axis=1)

# Write result
df = pd.read_csv(r'C:\hong\python-ws\project-1\result.csv')
df['class'] = preds
df.to_csv('DS2_challenge_team1_final.csv', index=False)
print("Final 5-Fold Ensemble completed: DS2_challenge_team1_final.csv")

# Print class distribution
counts = df['class'].value_counts().sort_index()
print("\n--- Final Test Class Distribution ---")
print(counts.to_dict())
