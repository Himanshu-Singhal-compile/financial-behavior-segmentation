"""Validate notebook structure and Python syntax without executing training."""
from pathlib import Path
import ast
import json

root = Path(__file__).resolve().parents[1]
count = 0
for path in (root / 'notebooks').glob('*.ipynb'):
    notebook = json.loads(path.read_text(encoding='utf-8'))
    assert notebook['nbformat'] == 4, path
    for index, cell in enumerate(notebook['cells']):
        if cell['cell_type'] != 'code':
            continue
        source = ''.join(cell['source'])
        ast.parse(source, filename=f'{path.name}:cell-{index}')
        assert not cell.get('outputs'), f'{path}: unexpected saved output'
        assert 'C:\\Users' not in source, f'{path}: machine-specific path'
    count += 1
print(f'Validated {count} notebooks (structure and syntax; training not executed).')
