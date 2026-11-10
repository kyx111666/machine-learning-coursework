"""Build source-only, self-contained Kaggle entry points for vessel segmentation."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'task4_vessel_segmentation'
MODULES = ['runtime', 'training', 'predictions', 'gamma', 'train', 'experiments']


def main():
    entries = [
        ('train_gamma', 'Single Gamma three-branch fusion', 'train', '--route', 'gamma'),
        ('train_seed_blend', 'Two RGB seeds and green-feature fusion', 'train', '--route', 'seed_blend'),
        ('experiments_screening', 'Model, hyperparameter and ablation screening', 'experiments', '--group', 'screening'),
        ('experiments_gamma', 'Paired Gamma augmentation experiment', 'experiments', '--group', 'gamma'),
    ]
    for name, title, script, flag, selection in entries:
        cells = []
        output = '/kaggle/working/task4' if script == 'train' else '/kaggle/working/task4_experiments'

        def markdown(source):
            cells.append(dict(cell_type='markdown', metadata={}, source=source.splitlines(keepends=True)))

        def code(source):
            cells.append(dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                              source=source.splitlines(keepends=True)))

        markdown(f'# Task 4: {title}\n\nAttach only the official 2026task-4 competition data and enable GPU. '
                 'No pretrained weights, previous Notebook Outputs or external datasets are required. '
                 'Internet is needed only if a dependency must be installed.\n\n'
                 f'Training and submission files are written under `{output}/`. '
                 'Keep the complete Notebook Output to retain fold checkpoints and probability files.')
        code('from pathlib import Path\nimport torch\n'
             'if not Path("/kaggle/working").is_dir() or not torch.cuda.is_available():\n'
             '    raise RuntimeError("Run this notebook on Kaggle with GPU enabled.")\n')
        code('%pip install -q "numpy>=1.26" "pandas>=2.2" "Pillow>=10" "scipy>=1.12" '
             '"scikit-learn>=1.4" "opencv-python-headless>=4.8"\n')
        code('Path("/kaggle/working/task4_source").mkdir(exist_ok=True)\n')
        for module in MODULES:
            code(f'%%writefile /kaggle/working/task4_source/{module}.py\n'+(ROOT/f'{module}.py').read_text())
        code('import subprocess\nimport sys\n'
             f'subprocess.run([sys.executable, "/kaggle/working/task4_source/{script}.py", '
             f'"--data", "/kaggle/input", "--out", "{output}", "{flag}", "{selection}"], check=True)\n')
        for i, cell in enumerate(cells):
            cell['id'] = f'{name}-{i}'
        nb = dict(cells=cells, metadata=dict(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                                           language_info=dict(name='python')), nbformat=4, nbformat_minor=5)
        (ROOT/'notebooks'/f'{name}.ipynb').write_text(json.dumps(nb, ensure_ascii=False, indent=1)+'\n')


if __name__ == '__main__':
    main()
