"""Build source-only Kaggle notebooks for the final fovea models and completed experiments."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'task3_fovea_localization'
CASCADE = ['data', 'models', 'training', 'submission', 'stages', 'inference', 'cascade', 'recipes', 'train']
COMPONENTS = ['data', 'submission', 'classic', 'train_components']


def main():
    entries = [
        ('train_cascade', 'Coarse-to-fine ensemble', CASCADE, 'train', 'final'),
        ('train_components', 'Dropout heatmaps and ImageNet ResNet34 components', COMPONENTS, 'train_components', 'final'),
        ('experiments_cascade', 'Coarse/fine ablations, hyperparameters and model comparisons', CASCADE, 'train', 'experiments'),
        ('experiments_components', 'Scratch and transfer controlled experiments', COMPONENTS, 'train_components', 'all'),
    ]
    for name, title, modules, script, group in entries:
        cells = []
        def md(text):
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
        def code(text):
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': text.splitlines(keepends=True)})
        md(f'# Task 3: {title}\n\nAdd only the official 2026task3 data. Enable GPU and Internet for ImageNet initialization.\n'
           'Training outputs are saved under /kaggle/working/task3. Collect the two final training outputs into one directory '
           'before running the fixed coordinate blends described in the README.')
        code('%pip install -q timm pandas Pillow scikit-learn opencv-python-headless\n')
        code('from pathlib import Path\nPath("/kaggle/working/task3_source").mkdir(exist_ok=True)\n')
        for module in modules:
            code(f'%%writefile /kaggle/working/task3_source/{module}.py\n' + (ROOT/f'{module}.py').read_text())
        code('import subprocess\n'+f'GROUP = {group!r}\n'+
             f'subprocess.run(["python", "/kaggle/working/task3_source/{script}.py", "--data", "/kaggle/input", '
             '"--out", "/kaggle/working/task3", "--group", GROUP], check=True)\n')
        for i, cell in enumerate(cells): cell['id'] = f'{name}-{i}'
        nb = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                                          'language_info': {'name': 'python'}}, 'nbformat': 4, 'nbformat_minor': 5}
        (ROOT/'notebooks'/f'{name}.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n')


if __name__ == '__main__': main()
