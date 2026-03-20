import ast
import json

with open('generate_notebook.py', encoding='utf-8') as f:
    exec(f.read())

with open('run_notebook_extracted.py', 'w', encoding='utf-8') as f:
    for c in cells:
        if c['cell_type'] == 'code':
            src = c['source']
            if isinstance(src, list):
                f.write(''.join(src))
            else:
                f.write(src)
            f.write('\n')
