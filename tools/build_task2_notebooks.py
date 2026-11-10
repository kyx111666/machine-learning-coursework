"""Build self-contained task 2 Kaggle notebooks from the released training sources."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'task2_emotion_classification'


def main():
    entries = [
        ('train_pretrained', 'Pretrained model families', 'train_pretrained', 'cv_resnet18'),
        ('train_resnet32', 'Scratch ResNet32 grouped cross-validation', 'train_resnet32', 'cv_resnet32'),
        ('train_single_resnet32', 'Scratch ResNet32 original holdout split', 'train_holdout', 'single_resnet32'),
        ('ablation_initial', 'Initial-recipe ablations', 'ablation_initial', 'ablation'),
        ('ablation_final', 'Final-recipe ablations', 'train_pretrained', 'final_ablation'),
        ('model_comparison', 'ResNet50 transfer-learning comparison', 'train_model_comparison', 'cv_resnet50'),
    ]
    for name, title, script, run in entries:
        cells = []
        def markdown(text):
            cells.append({'cell_type': 'markdown', 'metadata': {}, 'source': text.splitlines(keepends=True)})
        def code(text):
            cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                          'outputs': [], 'source': text.splitlines(keepends=True)})
        markdown(f'# Task 2: {title}\n\nAdd only the official 2026task2 competition images. Enable GPU.\n'
                 'Enable Internet for ImageNet-initialized models. Set RUN in the last cell to select the experiment.\n'
                 'Use the README to collect the required model outputs before final blending.')
        code('%pip install -q timm pandas Pillow scikit-learn matplotlib\n')
        code('from pathlib import Path\nPath("/kaggle/working/task2_source").mkdir(exist_ok=True)\n')
        for module in ['cli', script]:
            code(f'%%writefile /kaggle/working/task2_source/{module}.py\n' + (ROOT / f'{module}.py').read_text())
        code(f'import subprocess\nRUN = {run!r}\n'
             f'subprocess.run(["python", "/kaggle/working/task2_source/{script}.py", "--run", RUN, '
             '"--input", "/kaggle/input", "--out", "/kaggle/working/task2"], check=True)\n')
        for i, cell in enumerate(cells): cell['id'] = f'{name}-{i}'
        notebook = {'cells': cells, 'metadata': {'kernelspec': {'display_name': 'Python 3',
                    'language': 'python', 'name': 'python3'}, 'language_info': {'name': 'python'}},
                    'nbformat': 4, 'nbformat_minor': 5}
        (ROOT / 'notebooks' / f'{name}.ipynb').write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')


if __name__ == '__main__':
    main()
