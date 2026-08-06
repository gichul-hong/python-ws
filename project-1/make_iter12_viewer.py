import numpy as np
import pandas as pd
import json
import os

# Load probabilities from iter12
conv_test = np.load('iter12-gemini/conv_test_probs.npy')
scratch_test = np.load('iter12-gemini/scratch_test_probs.npy')

# iter12 optimized weights
opt_w = 0.4387
final_test_probs = opt_w * conv_test + (1 - opt_w) * scratch_test

preds = np.argmax(final_test_probs, axis=1)
confs = np.max(final_test_probs, axis=1)

# Get filenames
df = pd.read_csv('result.csv')
filenames = df['id'].values

data = []
for i in range(len(filenames)):
    data.append({
        'id': filenames[i],
        'class': int(preds[i]),
        'conf': float(confs[i]),
        'path': f'dataset/data 2/Test/{filenames[i]}'
    })

# HTML template
html_template = """<!DOCTYPE html>
<html>
<head>
    <title>Iter12 Confidence & Class Viewer</title>
    <style>
        body { font-family: sans-serif; background: #121212; color: #fff; margin: 20px; }
        .controls { margin-bottom: 20px; padding: 15px; background: #1e1e1e; border-radius: 8px; }
        select, button { padding: 8px; margin-right: 10px; background: #333; color: white; border: 1px solid #555; }
        .grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(150px, 1fr)); gap: 15px; }
        .card { background: #222; border-radius: 8px; padding: 10px; text-align: center; }
        .card img { max-width: 100%; height: auto; border-radius: 4px; }
        .conf { font-weight: bold; color: #4ade80; margin-top: 5px; }
        .low-conf { color: #f87171; }
    </style>
</head>
<body>
    <h2>Iter12 Confidence & Class Viewer</h2>
    <div class="controls">
        <label>Class Filter: </label>
        <select id="classFilter" onchange="render()">
            <option value="all">All Classes</option>
            <!-- Options will be populated by JS -->
        </select>
        <label>Sort By: </label>
        <select id="sortFilter" onchange="render()">
            <option value="conf_asc">Confidence (Low to High)</option>
            <option value="conf_desc">Confidence (High to Low)</option>
        </select>
        <span style="margin-left:20px;">Total items shown: <span id="count">0</span></span>
    </div>
    <div class="grid" id="grid"></div>

    <script>
        const data = DATA_PLACEHOLDER;
        
        // Populate class dropdown
        const classSelect = document.getElementById('classFilter');
        for(let i=0; i<43; i++) {
            const opt = document.createElement('option');
            opt.value = i;
            opt.innerText = `Class ${i}`;
            classSelect.appendChild(opt);
        }

        function render() {
            const classVal = classSelect.value;
            const sortVal = document.getElementById('sortFilter').value;
            
            let filtered = data;
            if(classVal !== 'all') {
                filtered = filtered.filter(d => d.class == parseInt(classVal));
            }
            
            if(sortVal === 'conf_asc') {
                filtered.sort((a,b) => a.conf - b.conf);
            } else {
                filtered.sort((a,b) => b.conf - a.conf);
            }
            
            document.getElementById('count').innerText = filtered.length;
            
            const grid = document.getElementById('grid');
            grid.innerHTML = '';
            
            // Limit to 500 for performance
            const displayData = filtered.slice(0, 500);
            
            displayData.forEach(d => {
                const confClass = d.conf < 0.9 ? 'low-conf' : '';
                const card = document.createElement('div');
                card.className = 'card';
                card.innerHTML = `
                    <img src="${d.path}" loading="lazy" onerror="this.src=''" alt="${d.id}">
                    <div style="margin-top:8px;">${d.id}</div>
                    <div style="color:#aaa;">Class ${d.class}</div>
                    <div class="conf ${confClass}">${(d.conf * 100).toFixed(2)}%</div>
                `;
                grid.appendChild(card);
            });
        }
        
        // Initial render
        render();
    </script>
</body>
</html>
"""

html_out = html_template.replace("DATA_PLACEHOLDER", json.dumps(data))
with open('iter12_viewer.html', 'w', encoding='utf-8') as f:
    f.write(html_out)

print("iter12_viewer.html has been created successfully!")
