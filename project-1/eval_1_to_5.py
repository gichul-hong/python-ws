import pandas as pd
import glob
from sklearn.metrics import accuracy_score, f1_score
import os

answer_df = pd.read_excel(r'C:\hong\python-ws\project-1\answer.xlsx')
answer_df = answer_df.rename(columns={'정답': 'id', '정답.1': 'true_class'})

iters = ['iter1', 'iter2-fable', 'iter3-fable', 'iter4-zai', 'iter5-zai']
results = []
for it in iters:
    csv_files = glob.glob(rf'C:\hong\python-ws\project-1\{it}\*.csv')
    csv_files = [f for f in csv_files if 'team1' in f or 'result' in f]
    
    if not csv_files:
        print(f'{it}: No prediction CSV found')
        continue
    
    target_csv = None
    for f in csv_files:
        if 'final' in f:
            target_csv = f
            break
    if not target_csv:
        target_csv = csv_files[0]
        
    try:
        pred_df = pd.read_csv(target_csv)
        if 'class' in pred_df.columns:
            pred_df = pred_df.rename(columns={'class': 'pred_class'})
        else:
            pred_df = pred_df.rename(columns={pred_df.columns[1]: 'pred_class'})
            
        merged = pd.merge(pred_df, answer_df, on='id', how='inner')
        y_true = merged['true_class'].values
        y_pred = merged['pred_class'].values
        
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, average='macro')
        results.append(f'{it}: Acc = {acc:.4f}, Macro F1 = {f1:.4f} (File: {os.path.basename(target_csv)})')
    except Exception as e:
        results.append(f'{it}: Error evaluating {os.path.basename(target_csv)} - {e}')

print('\n'.join(results))
