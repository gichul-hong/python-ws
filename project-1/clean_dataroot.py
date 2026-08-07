import json

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb.get('cells', [])):
    if cell.get('cell_type') == 'code' and i > 0:  # Skip the first setup cell!
        new_source = []
        for line in cell['source']:
            # If the line defines DATA_ROOT, skip adding it
            if line.startswith('DATA_ROOT =') or line.strip() == "DATA_ROOT = '/content/drive/MyDrive/DS2026/20260803-pjt1-강유/조별과제/dataset/data 2'":
                continue
            new_source.append(line)
        cell['source'] = new_source

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Cleaned up duplicate DATA_ROOT in lower cells!')
