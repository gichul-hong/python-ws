import json

user_path = "'/content/drive/MyDrive/DS2026/20260803-pjt1-강유/조별과제/dataset/data 2'"

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            if line.startswith('DATA_ROOT =') and 'drive.mount' not in line:
                line = f'DATA_ROOT = {user_path}\n'
            new_source.append(line)
        cell['source'] = new_source

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated DATA_ROOT in all cells!')
