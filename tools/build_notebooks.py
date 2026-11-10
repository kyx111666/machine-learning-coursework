"""Build clean, self-contained Kaggle notebooks from the released source files."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'task1_plant_classification'


def markdown(text):
    return {'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)}


def code(text):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': text.splitlines(keepends=True)}


def main():
    entries = {
        'train_convnext': ('ConvNeXt-Tiny 五折训练', ['data', 'train_convnext'], '添加 2026task1 竞赛数据。'),
        'train_dinov2': ('DINOv2-Base 五折微调与融合', ['data', 'train_dinov2'],
                         '添加竞赛数据及第一步的 convnext_probabilities.npz。'),
        'train_ensemble': ('DINOv2 全数据三种子与最终融合', ['data', 'train_dinov2', 'train_ensemble'],
                           '添加竞赛数据、convnext_probabilities.npz、dinov2_cv_probabilities.npz。'),
        'ablation': ('植物分类消融实验', ['data', 'ablation'], '添加 2026task1 竞赛数据，最后一格选择所需实验。'),
    }
    for entry, (title, modules, inputs) in entries.items():
        cells = [markdown(f'# {title}\n\n{inputs}\n\n开启 GPU 和 Internet 后按顺序运行。'),
                 code('%pip install -q timm scipy scikit-learn pandas Pillow\n'),
                 code('from pathlib import Path\nPath("/kaggle/working/task1_source").mkdir(exist_ok=True)\n')]
        for module in modules:
            cells.append(code(f'%%writefile /kaggle/working/task1_source/{module}.py\n' +
                              (ROOT / f'{module}.py').read_text()))
        arguments = ['--data', '/kaggle/input', '--out', '/kaggle/working']
        if entry == 'ablation':
            arguments = ['--data', '/kaggle/input', '--out', '/kaggle/working/results', '--exp', 'base']
        command = ['python', f'/kaggle/working/task1_source/{entry}.py', *arguments]
        cells.append(code('import subprocess\nsubprocess.run(' + repr(command) + ', check=True)\n'))
        for index, cell in enumerate(cells):
            cell['id'] = f'{entry}-{index}'
        notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                    'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
                    'nbformat': 4, 'nbformat_minor': 5}
        (ROOT / 'notebooks' / f'{entry}.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
