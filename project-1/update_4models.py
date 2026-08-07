import json
import os

# 1. Update ensemble_4models.py
with open(r'C:\hong\python-ws\project-1\iter18-final\ensemble_5models.py', 'r', encoding='utf-8') as f:
    ens_content = f.read()
ens_content = ens_content.replace("['conv', 'scratch', 'resnet', 'efficientnet', 'mobilenet']", "['conv', 'resnet', 'efficientnet', 'mobilenet']")
ens_content = ens_content.replace("x0=[0.2]*5", "x0=[0.25]*4")
ens_content = ens_content.replace("bounds=[(0.0, 1.0)]*5", "bounds=[(0.0, 1.0)]*4")
ens_content = ens_content.replace("5-Model", "4-Model")

with open(r'C:\hong\python-ws\project-1\iter18-final\ensemble_4models.py', 'w', encoding='utf-8') as f:
    f.write(ens_content)
    
if os.path.exists(r'C:\hong\python-ws\project-1\iter18-final\ensemble_5models.py'):
    os.remove(r'C:\hong\python-ws\project-1\iter18-final\ensemble_5models.py')

# 2. Update run_all.ps1
with open(r'C:\hong\python-ws\project-1\iter18-final\run_all.ps1', 'r', encoding='utf-8') as f:
    run_content = f.read()
run_content = run_content.replace('conda run -n gpu-torch --no-capture-output python train_scratch.py\n', '')
run_content = run_content.replace('ensemble_5models.py', 'ensemble_4models.py')
run_content = run_content.replace('5-Model', '4-Model')

with open(r'C:\hong\python-ws\project-1\iter18-final\run_all.ps1', 'w', encoding='utf-8') as f:
    f.write(run_content)

# 3. Update iter18_colab.ipynb
with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []
for cell in nb.get('cells', []):
    source = ''.join(cell.get('source', []))
    if 'train_scratch.py' in source:
        continue # Remove scratch cell
    if 'ensemble_5models' in source:
        source = source.replace("['conv', 'scratch', 'resnet', 'efficientnet', 'mobilenet']", "['conv', 'resnet', 'efficientnet', 'mobilenet']")
        source = source.replace("x0=[0.2]*5", "x0=[0.25]*4")
        source = source.replace("bounds=[(0.0, 1.0)]*5", "bounds=[(0.0, 1.0)]*4")
        source = source.replace("5-Model", "4-Model")
        cell['source'] = [line + '\n' for line in source.split('\n') if line.strip() != '']
    new_cells.append(cell)
nb['cells'] = new_cells

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated everything for 4 models (No Scratch)')
