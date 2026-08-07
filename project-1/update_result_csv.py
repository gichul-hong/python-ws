import json

result_csv_path = "'/content/drive/MyDrive/DS2026/20260803-pjt1-강유/조별과제/result.csv'"

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    if cell.get('cell_type') == 'code':
        new_source = []
        for line in cell['source']:
            # Replace './result.csv' with the actual drive path
            if "'./result.csv'" in line:
                line = line.replace("'./result.csv'", result_csv_path)
            new_source.append(line)
        cell['source'] = new_source

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Updated result.csv path in notebook!')
