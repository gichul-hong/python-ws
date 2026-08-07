import json

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb.get('cells', []):
    source = ''.join(cell.get('source', []))
    if 'ResNetWithFeatures' in source:
        source = source.replace('conv_oof_probs.npy', 'resnet_oof_probs.npy')
        source = source.replace('conv_test_probs.npy', 'resnet_test_probs.npy')
    elif 'EfficientNetWithFeatures' in source:
        source = source.replace('conv_oof_probs.npy', 'efficientnet_oof_probs.npy')
        source = source.replace('conv_test_probs.npy', 'efficientnet_test_probs.npy')
    elif 'MobileNetWithFeatures' in source:
        source = source.replace('conv_oof_probs.npy', 'mobilenet_oof_probs.npy')
        source = source.replace('conv_test_probs.npy', 'mobilenet_test_probs.npy')
    
    cell['source'] = [line + '\n' for line in source.split('\n') if line.strip() != '' or line == '']

with open(r'C:\hong\python-ws\project-1\iter18_colab.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print('Fixed notebook filenames!')
