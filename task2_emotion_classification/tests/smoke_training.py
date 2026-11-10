"""Run short CPU training and checkpoint-reuse checks with generated images."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
LABELS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def main():
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp)
        data = work / 'input' / '2026task2'
        rng = np.random.default_rng(18)
        for label, name in enumerate(LABELS):
            folder = data / 'train' / name
            folder.mkdir(parents=True)
            for number in range(40):
                image = rng.integers(25, 85, (48, 48), dtype=np.uint8)
                image[2 + label * 7:7 + label * 7, 8:40] = 220
                Image.fromarray(image).save(folder / f'{name}_{number:03d}.png')
        test = data / 'test'
        test.mkdir()
        ids = [f'{number:04d}.png' for number in range(6)]
        for name in ids:
            Image.fromarray(rng.integers(0, 256, (48, 48), dtype=np.uint8)).save(test / name)
        pd.DataFrame({'ID': ids, 'Emotion': 0}).to_csv(data / 'sample_submission.csv', index=False)
        env = {**os.environ, 'TASK2_SMOKE_EPOCHS': '1', 'TASK2_SMOKE_SIZE': '32',
               'TASK2_SMOKE_EXPS': '1', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
               'MPLCONFIGDIR': str(work / 'matplotlib')}
        experiments = [('train_resnet32.py', 'cv_resnet32'),
                       ('train_pretrained.py', 'cv_resnet18'),
                       ('train_pretrained.py', 'cv_vgg_scratch48'),
                       ('ablation_initial.py', 'ablation'),
                       ('train_model_comparison.py', 'cv_resnet50'),
                       ('train_holdout.py', 'single_resnet32')]
        for script, run in experiments:
            out = work / f'out_{run}'
            args = [sys.executable, str(ROOT / script), '--run', run, '--input', str(work / 'input'),
                    '--out', str(out), '--smoke']
            if script != 'train_holdout.py':
                args += ['--folds', '0']
            run_env = {**env, 'TASK2_SMOKE_SIZE': '48' if script == 'train_resnet32.py' else '32'}
            result = subprocess.run(args, env=run_env, capture_output=True, text=True, timeout=180)
            if result.returncode:
                raise RuntimeError(f'{script}/{run}\n{result.stdout[-1500:]}\n{result.stderr[-3000:]}')
            if script == 'train_holdout.py':
                folder = out / 'smoke_run' / 'single_resnet32'
                probabilities = np.load(folder / 'test_probabilities.npy')
                assert probabilities.shape == (6, 6)
                assert pd.read_csv(folder / 'submission.csv').ID.tolist() == ids
            else:
                folder = out / run
                status = json.loads((folder / 'status.json').read_text())
                assert not status['pending']
                files = list(folder.glob('oof_*.npz'))
                assert len(files) == 1
                with np.load(files[0]) as saved:
                    mask = saved['mask']
                    assert 0 < mask.sum() < len(mask)
                    np.testing.assert_allclose(saved['ema_flip'][mask].sum(1), 1, atol=1e-4)
                if run == 'cv_resnet32':
                    before = files[0].read_bytes()
                    again = subprocess.run(args, env=run_env, capture_output=True, text=True, timeout=180)
                    assert again.returncode == 0, again.stderr[-3000:]
                    assert '已完成，复用结果' in again.stdout
                    assert files[0].read_bytes() == before
            print(f'{script}: {run} passed', flush=True)
        analysis = subprocess.run([sys.executable, str(ROOT / 'analyze_results.py'), '--results',
                                   str(work / 'out_ablation'), '--out', str(work / 'analysis')],
                                  env=env, capture_output=True, text=True, timeout=180)
        if analysis.returncode:
            raise RuntimeError(analysis.stderr[-3000:])
        assert (work / 'analysis' / 'ablation_comparison.csv').is_file()
        assert (work / 'analysis' / 'ablation_initial.png').is_file()
        print('Analysis tables and figures passed', flush=True)


if __name__ == '__main__':
    main()
