"""Grouped-fold workflow checks with generated images and miniature networks; no model downloads."""
import tempfile
from pathlib import Path
from unittest.mock import patch
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import pandas as pd
import torch
from torch import nn
import training
import inference
import classic
import train
from train import train_recipes, export_cascade
from train_components import run_component, SCRATCH
from blend import FORMULAS, main as blend_main


class TinyHeatNet(nn.Module):
    stride = 4
    def __init__(self):
        super().__init__()
        self.enc = nn.Conv2d(3, 4, 4, stride=4)
        self.head = nn.Conv2d(4, 1, 1)
    def forward(self, image): return self.head(torch.relu(self.enc(image))).squeeze(1)


class TinyClassic(nn.Module):
    def __init__(self, transfer=False):
        super().__init__()
        setattr(self, 'encoder' if transfer else 'features', nn.Conv2d(3, 4, 3, padding=1))
        self.head = nn.Conv2d(4, 1, 4, stride=4)
        self.visibility = nn.Linear(4, 1)
    def forward(self, image):
        features = getattr(self, 'encoder', getattr(self, 'features', None))(image)
        logits = self.head(torch.relu(features)).squeeze(1)
        return classic.decode_heatmap(logits, 'global'), self.visibility(features.mean((2, 3))).squeeze(1), logits


def make_data(root):
    (root/'train').mkdir(parents=True); (root/'test').mkdir()
    rng = np.random.default_rng(2026); records = []
    for i in range(1, 101):
        image = np.zeros((96, 128, 3), np.uint8)
        circle = np.zeros((96, 128), np.uint8)
        cv2.circle(circle, (64, 48), 40, 1, -1)
        image[circle.astype(bool)] = rng.integers(50, 180, (int(circle.sum()), 3), dtype=np.uint8)
        cv2.imwrite(str(root/('train' if i <= 80 else 'test')/f'{i:04d}.jpg'), image)
        if i <= 80: records.append({'data': f'{i:04d}', 'Fovea_X': 63+i%3, 'Fovea_Y': 47+i%3})
    (root/'train/0077.jpg').write_bytes((root/'train/0039.jpg').read_bytes())
    pd.DataFrame(records).to_csv(root/'fovea_localization_train_GT.csv', index=False)
    pd.DataFrame({'ImageID': [f'{i:04d}_Fovea_{axis}' for i in range(81,101) for axis in 'XY'],
                  'value': 0}).to_csv(root/'sample_submission.csv', index=False)


def main():
    torch.set_num_threads(1)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp); data = work/'data'; make_data(data); out = work/'outputs'
        names = ['c_r34_256', 'f640_wide_s600']
        overrides = {
            names[0]: dict(encoder='resnet18', pretrained=False, size=64, sigma=2., steps=2,
                           batch=2, workers=1, val_every=0, save_weights=True),
            names[1]: dict(encoder='resnet18', pretrained=False, size=64, crop=48, sigma=2.,
                           steps=2, batch=2, workers=1, val_every=0, save_weights=True),
        }
        with patch.object(training, 'build_model', lambda *args, **kwargs: TinyHeatNet()):
            items, tests, sample = train_recipes(names, data, out, torch.device('cpu'), config_overrides=overrides)
            with np.load(out/'runs'/names[1]/'oof.npz') as saved:
                assert saved['p15|flip|it1'].shape == (80,2) and np.isfinite(saved['p15|flip|it1']).all()
            before = (out/'runs'/names[0]/'fold0.npz').read_bytes()
            train_recipes(names, data, out, torch.device('cpu'), config_overrides=overrides)
            assert (out/'runs'/names[0]/'fold0.npz').read_bytes() == before
            changed = {**overrides, names[0]: {**overrides[names[0]], 'sigma': 3.}}
            try: train_recipes([names[0]], data, out, torch.device('cpu'), config_overrides=changed)
            except ValueError: pass
            else: raise AssertionError('Changed configuration reused a completed fold')
        with (patch.object(inference, 'build_model', lambda *args, **kwargs: TinyHeatNet()),
              patch.object(train, 'FINAL_COARSE', names[:1]), patch.object(train, 'FINAL_FINE', names[1:])):
            result = export_cascade(out, items, tests, sample, torch.device('cpu'), data)
            assert result['oof'].shape == (80, 2) and np.isfinite(result['test']).all()
        cfg = dict(SCRATCH, width=4, size=32, epochs=10, repeat=1, batch=16)
        with patch.object(classic, 'Locator', lambda width, model: TinyClassic(model.startswith('resnet'))):
            run_component('scratch', cfg, 1, data, out/'components', torch.device('cpu'))
            with np.load(out/'components/scratch/predictions_sharp.npz', allow_pickle=False) as saved:
                assert saved['oof'].shape == (80,2) and saved['test'].shape == (20,2)
                assert np.isfinite(saved['test']).all()
                first_predictions = {key: saved[key] for key in saved.files}
            state = torch.load(out/'components/scratch/fold0/last.pt', weights_only=False)
            assert state['n_average'] == 10
            before = (out/'components/scratch/fold0/model.pt').read_bytes()
            run_component('scratch', cfg, 1, data, out/'components', torch.device('cpu'))
            assert (out/'components/scratch/fold0/model.pt').read_bytes() == before
            run_component('scratch_second', cfg, 2, data, out/'components', torch.device('cpu'))
            with np.load(out/'components/scratch_second/predictions_sharp.npz') as second:
                from data import SPLIT2_FOLDS
                assert second['folds'].tolist() == [int(SPLIT2_FOLDS[Path(name).stem]) for name in second['train_names']]
                second_predictions = {key: second[key] for key in second.files}
            transfer = dict(cfg, model='resnet34_scratch', lr=1e-3, encoder_lr=1e-4)
            run_component('transfer', transfer, 1, data, out/'components', torch.device('cpu'), folds=[0])
            checkpoint = torch.load(out/'components/transfer/fold0/last.pt', weights_only=False)
            groups = checkpoint['opt']['param_groups']
            assert len(groups) == 2 and abs(groups[1]['lr']/groups[0]['lr']-10) < 1e-6
        for name, predictions in [('dropout_split1', first_predictions), ('dropout_split2', second_predictions),
                                  ('resnet34_transfer', first_predictions)]:
            directory = out/'components'/name
            directory.mkdir()
            np.savez_compressed(directory/'predictions_sharp.npz', **predictions)
        for recipe in FORMULAS:
            blend_main(['--data', str(data), '--results', str(out), '--recipe', recipe,
                        '--out', str(work/'submissions')])
            saved = pd.read_csv(work/'submissions'/f'submission_{recipe}.csv')
            assert saved.ImageID.tolist() == sample.ImageID.tolist() and np.isfinite(saved.value).all()
        print('Five-fold coarse/fine training, cascade export, both grouped scratch splits, transfer optimization, '
              'both final blends and cache checks passed.')


if __name__ == '__main__': main()
