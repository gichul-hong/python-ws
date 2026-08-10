import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

answer_df = pd.read_excel(r'C:\hong\python-ws\project-1\answer.xlsx')
answer_df = answer_df.rename(columns={'정답': 'id', '정답.1': 'true_class'})

pred_df = pd.read_csv(r'C:\hong\python-ws\project-1\iter19-hero\hero_result.csv')
if 'class' in pred_df.columns:
    pred_df = pred_df.rename(columns={'class': 'pred_class'})
else:
    pred_df = pred_df.rename(columns={pred_df.columns[1]: 'pred_class'})
    
merged = pd.merge(pred_df, answer_df, on='id', how='inner')
y_true = merged['true_class'].values
y_pred = merged['pred_class'].values

acc = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='macro')

print(f'iter19-hero: Acc = {acc:.6f}, Macro F1 = {f1:.6f}')
