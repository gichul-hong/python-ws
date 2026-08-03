import os
import numpy as np
import pandas as pd

train = r'C:\hong\python-ws\project-1\dataset\data 2\Train'
tc = np.array([len([f for f in os.listdir(os.path.join(train, str(i))) if f.endswith('.png')]) for i in range(43)])
exp = tc / 3.0  # PDF test chart pattern = train counts / 3 (180->60, 630->210, 1260->420)
pred = pd.read_csv(r'C:\hong\python-ws\project-1\iter2-fable\result.csv')['class'] \
    .value_counts().reindex(range(43), fill_value=0).values

print('total pred:', pred.sum(), ' total expected:', exp.sum())
diff = pred - exp
print(f'{"cls":>3} {"exp":>6} {"pred":>5} {"diff":>6}')
for i in range(43):
    print(f'{i:>3} {exp[i]:>6.0f} {pred[i]:>5} {diff[i]:>+6.0f}')
print()
print('sum |diff|:', np.abs(diff).sum(), '-> mismatch rate:', np.abs(diff).sum() / pred.sum())
print('max |diff|:', np.abs(diff).max())
print('pearson corr:', np.corrcoef(exp, pred)[0, 1])
print('chi-square:', ((pred - exp) ** 2 / exp).sum())
