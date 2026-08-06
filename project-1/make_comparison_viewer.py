import numpy as np
import pandas as pd
import json

# Load iter12 probabilities
conv_test = np.load('iter12-gemini/conv_test_probs.npy')
scratch_test = np.load('iter12-gemini/scratch_test_probs.npy')

# iter12 optimized weights
opt_w = 0.4387
iter12_probs = opt_w * conv_test + (1 - opt_w) * scratch_test
iter12_preds = np.argmax(iter12_probs, axis=1)
iter12_confs = np.max(iter12_probs, axis=1)

# Load CS preds
df_cs = pd.read_csv('result_test_convnext160_tta_blend.csv')
cs_preds = df_cs.iloc[:, -1].values
filenames = df_cs['id'].values

data = []
for i in range(len(filenames)):
    if iter12_preds[i] != cs_preds[i]:
        data.append({
            'id': filenames[i],
            'cs_class': int(cs_preds[i]),
            'iter12_class': int(iter12_preds[i]),
            'iter12_conf': float(iter12_confs[i]),
            'path': f'dataset/data 2/Test/{filenames[i]}'
        })

# HTML template
html_template = """<!DOCTYPE html>
<html>
<head>
    <title>CS vs Iter12 Comparison Viewer</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: #fff; margin: 20px; }
        .controls { margin-bottom: 20px; padding: 15px; background: #1e1e1e; border-radius: 8px; }
        select, button { padding: 8px; margin-right: 10px; background: #333; color: white; border: 1px solid #555; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr)); gap: 15px; }
        .card { background: #222; border-radius: 8px; padding: 15px; text-align: center; border: 1px solid #444; }
        .card img { max-width: 100%; height: auto; border-radius: 4px; margin-bottom: 10px; }
        .label { font-size: 14px; margin: 4px 0; }
        .cs-pred { color: #4ade80; font-weight: bold; }
        .us-pred { color: #f87171; font-weight: bold; }
        .conf { font-size: 12px; color: #aaa; }
    </style>
</head>
<body>
    <h2>CS Model vs Iter12 Differences</h2>
    <div class="controls">
        <label>Filter by Iter12's Class: </label>
        <select id="classFilter" onchange="render()">
            <option value="all">All Differences</option>
        </select>
        <span style="margin-left:20px;">Differences found: <span id="count">0</span></span>
    </div>
    <div class="grid" id="grid"></div>

    <script>
        const data = DATA_PLACEHOLDER;
        
        // Populate class dropdown based on existing diffs
        const classSelect = document.getElementById('classFilter');
        const uniqueClasses = [...new Set(data.map(d => d.iter12_class))].sort((a,b)=>a-b);
        uniqueClasses.forEach(c => {
            const opt = document.createElement('option');
            opt.value = c;
            opt.innerText = `Iter12 predicted Class ${c}`;
            classSelect.appendChild(opt);
        });

        function render() {
            const classVal = classSelect.value;
            let filtered = data;
            
            if(classVal !== 'all') {
                filtered = filtered.filter(d => d.iter12_class == parseInt(classVal));
            }
            
            // Sort by iter12 confidence (lowest first)
            filtered.sort((a,b) => a.iter12_conf - b.iter12_conf);
            
            document.getElementById('count').innerText = filtered.length;
            
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            
            filtered.forEach(d => {
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <img src="${d.path}" loading="lazy" onerror="this.src=''" alt="${d.id}">
                    <div style="font-size: 12px; color: #888; margin-bottom: 8px;">${d.id}</div>
                    <div class="label cs-pred">CS Model: Class ${d.cs_class}</div>
                    <div class="label us-pred">Iter12: Class ${d.iter12_class}</div>
                    <div class="conf">(Iter12 Conf: ${(d.iter12_conf * 100).toFixed(2)}%)</div>
                `;
                grid.appendChild(card);
            });
        }
        
        render();
    </script>
</body>
</html>
"""

html_out = html_template.replace("DATA_PLACEHOLDER", json.dumps(data))
with open('compare_cs_iter12.html', 'w', encoding='utf-8') as f:
    f.write(html_out)

print(f"Created compare_cs_iter12.html with {len(data)} differences.")
