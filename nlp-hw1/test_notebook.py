import json
with open('HW1-intent-classification.ipynb', 'r') as f:
    nb = json.load(f)
print(f'Total cells: {len(nb["cells"])}')
for i, cell in enumerate(nb['cells']):
    src = ''.join(cell['source'])[:100].replace('\n', ' ')
    print(f'Cell {i} [{cell["cell_type"]:8s}]: {src}')