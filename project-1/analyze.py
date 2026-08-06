import numpy as np
import pandas as pd

# Load labels and compute expected
oof_labels = np.load('iter11-gemini/oof_labels.npy')
tc = np.array([np.sum(oof_labels == i) for i in range(43)])
expected = (tc / len(oof_labels)) * 8670

# Load predictions
df_cs = pd.read_csv('result_test_convnext160_tta_blend.csv')
preds_cs = df_cs.iloc[:, -1].values

# using iter11 ensemble (or iter12 ensemble) for comparison, wait we don't have iter13 yet
df_us = pd.read_csv('iter12-gemini/conv_result.csv') # single ConvNeXt 5-fold as reference
preds_us = df_us.iloc[:, -1].values

pred_c_cs = np.array([np.sum(preds_cs == i) for i in range(43)])
pred_c_us = np.array([np.sum(preds_us == i) for i in range(43)])

diff_cs = pred_c_cs - expected
diff_us = pred_c_us - expected

print('\n--- Comparison (Top 10 differences by CS model) ---')
print('Class | Expected | CS Pred (Diff) | US Pred (Diff)')
indices = np.argsort(np.abs(diff_cs))[::-1][:10]
for i in indices:
    print(f'{i:5d} | {expected[i]:8.1f} | {pred_c_cs[i]:7d} ({diff_cs[i]:+5.1f}) | {pred_c_us[i]:7d} ({diff_us[i]:+5.1f})')

print('\n--- Classes where CS model fixed US model errors ---')
us_errors = np.argsort(np.abs(diff_us))[::-1]
count = 0
for i in us_errors:
    if abs(diff_us[i]) > 3 and abs(diff_cs[i]) < abs(diff_us[i]):
        print(f'Class {i:2d}: US diff {diff_us[i]:+5.1f} -> CS diff {diff_cs[i]:+5.1f}')
        count += 1
if count == 0:
    print("None found or no major fixes needed.")
