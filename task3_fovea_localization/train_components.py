"""Train the scratch and torchvision ResNet members and their completed controlled comparisons."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import classic
from submission import write_submission
from data import load_dataset, find_data_dir, dataset_fingerprint

DEFAULT = dict(model='heatmap', width=24, size=256, lr=1e-3, weight_decay=1e-4,
               crop=True, geometry=True, photo=True, clahe=False, sigma=8., epochs=100,
               repeat=2, batch=8, accumulate=1)
SCRATCH = dict(DEFAULT, model='heatmap_dropout', lr=3e-4, epochs=200)
TRANSFER = dict(DEFAULT, model='resnet34_imagenet', lr=1e-3, encoder_lr=1e-4, epochs=100)
COMPONENTS = {'dropout_split1': (SCRATCH, 1), 'dropout_split2': (SCRATCH, 2),
              'resnet34_transfer': (TRANSFER, 1)}
BASELINE = {
    'heatmap_main': DEFAULT,
    'coordinate_regression': dict(DEFAULT, model='regression'),
    'without_geometry': dict(DEFAULT, geometry=False),
    'without_photometric': dict(DEFAULT, photo=False),
    'without_retina_crop': dict(DEFAULT, crop=False),
    'heatmap_lr3e4': dict(DEFAULT, lr=3e-4),
    'gaussian_sigma4': dict(DEFAULT, sigma=4.),
    'with_clahe': dict(DEFAULT, clahe=True),
    'input384': dict(DEFAULT, size=384, sigma=12., batch=4, accumulate=2),
}
DURATION = {'heatmap_lr3e4': dict(DEFAULT, lr=3e-4),
            'duration_200': dict(DEFAULT, lr=3e-4, epochs=200),
            'dropout_100': dict(DEFAULT, model='heatmap_dropout', lr=3e-4),
            'dropout_split1': SCRATCH}
TRANSFER_COMPARISON = {
    'resnet18_scratch': dict(DEFAULT, model='resnet18_scratch', lr=1e-3, encoder_lr=3e-4),
    'resnet18_imagenet_lr3e4': dict(DEFAULT, model='resnet18_imagenet', lr=1e-3, encoder_lr=3e-4),
    'resnet18_imagenet_lr1e4': dict(DEFAULT, model='resnet18_imagenet', lr=1e-3, encoder_lr=1e-4),
    'resnet34_transfer': TRANSFER,
}
EXPERIMENTS = {**{name: (cfg, 1) for name, cfg in BASELINE.items()},
               **{name: (cfg, 1) for name, cfg in DURATION.items()},
               **{name: (cfg, 1) for name, cfg in TRANSFER_COMPARISON.items()}}


def export_component(name, cfg, split, base, frame, pixels, sample, root, device):
    directory = root / name
    test_frame, test_pixels = classic.test_pixels(base)
    oof = {mode: np.full((len(frame), 2), np.nan) for mode in classic.MODES}
    tests = {mode: [] for mode in classic.MODES}
    for fold in range(5):
        checkpoint = directory / f'fold{fold}' / 'model.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if state['cfg'] != cfg or state['fingerprint'] != classic.checkpoint_fingerprint(cfg, fold, frame):
            raise ValueError('Checkpoint data, configuration or grouped split differs')
        # Full model state replaces initialization; inference does not download ImageNet weights.
        model_name = cfg['model'].replace('_imagenet', '_scratch')
        model = classic.Locator(cfg['width'], model_name).to(device)
        model.load_state_dict(state['model'])
        valid = frame[frame.fold == fold]
        for items, image_cache, is_valid in [(valid, pixels, True), (test_frame, test_pixels, False)]:
            loader = DataLoader(classic.Fundus(items, image_cache, cfg), batch_size=cfg['batch'], num_workers=0)
            if cfg['model'] == 'regression':
                values = {'global': classic.predict(model, loader, device, cfg['size'])[0]}
            else:
                values = classic.predict_decoders(model, loader, device, cfg['size'])
            for mode, predictions in values.items():
                if is_valid:
                    oof[mode][valid.index] = predictions
                else:
                    tests[mode].append(predictions)
        del model
    modes = ['global'] if cfg['model'] == 'regression' else list(classic.MODES)
    metrics = []
    for mode in modes:
        predicted_test = np.mean(tests[mode], axis=0)
        if not np.isfinite(oof[mode]).all() or not np.isfinite(predicted_test).all():
            raise ValueError('Incomplete coordinate predictions')
        np.savez_compressed(directory / f'predictions_{mode}.npz', oof=oof[mode], test=predicted_test,
                            train_names=frame.image.to_numpy(dtype=str), test_names=test_frame.image.to_numpy(dtype=str),
                            target=frame[['x', 'y']].to_numpy(), folds=frame.fold.to_numpy(),
                            groups=frame.group.to_numpy(dtype=str), split=split,
                            dataset_fingerprint=dataset_fingerprint(base))
        metrics.append({'decoder': mode, 'rmse': classic.rmse(frame[['x', 'y']].to_numpy(), oof[mode])})
    pd.DataFrame(metrics).to_csv(directory / 'decoder_metrics.csv', index=False)
    chosen = 'global' if cfg['model'] == 'regression' else 'sharp'
    train, test, _ = load_dataset(base, split)
    if [item.name for item in test] != test_frame.image.tolist():
        raise ValueError('Test prediction order differs from official image order')
    write_submission(test, np.mean(tests[chosen], axis=0), sample, directory / 'submission.csv')
    return metrics


def run_component(name, cfg, split, data, output, device, folds=range(5)):
    cfg = copy.deepcopy(cfg)
    base, frame, pixels, sample = classic.load_official(data, split)
    directory = output / name
    directory.mkdir(parents=True, exist_ok=True)
    plan = {'config': cfg, 'split': split}
    path = directory / 'config.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError('Component configuration changed; use a new output directory')
    path.write_text(json.dumps(plan, indent=2))
    frame.to_csv(directory / 'training_manifest.csv', index=False)
    for fold in folds:
        classic.train_fold(name, cfg, fold, frame, pixels, output, device)
    if all((directory / f'fold{fold}' / 'complete.json').exists() for fold in range(5)):
        return export_component(name, cfg, split, base, frame, pixels, sample, output, device)
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--group', choices=['final', 'baseline', 'duration', 'transfer', 'all'], default='final')
    parser.add_argument('--run', choices=sorted(COMPONENTS.keys() | EXPERIMENTS.keys()))
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(sorted(COMPONENTS.keys() | EXPERIMENTS.keys())))
        return
    selection = dict(COMPONENTS) if args.group == 'final' else (
        {name: (cfg, 1) for name, cfg in {'baseline': BASELINE, 'duration': DURATION,
                                        'transfer': TRANSFER_COMPARISON}.get(args.group, {}).items()})
    if args.group == 'all':
        selection = {**COMPONENTS, **EXPERIMENTS}
    if args.run:
        selection = {args.run: {**COMPONENTS, **EXPERIMENTS}[args.run]}
    data = find_data_dir(args.data)
    root = args.out / 'components'
    for name, (config, split) in selection.items():
        print(name, run_component(name, config, split, data, root, torch.device(args.device), args.folds), flush=True)


if __name__ == '__main__':
    main()
