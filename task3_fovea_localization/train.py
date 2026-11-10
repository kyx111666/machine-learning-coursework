"""Train the fixed coarse-to-fine ensemble or the completed controlled experiments."""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data import load_dataset, find_data_dir, SPLIT2_FOLDS, dataset_fingerprint
from stages import run_coarse, run_fine, write_submission
from cascade import final_predictions
from training import default_device
from recipes import RECIPES, SECOND_SPLIT, FINAL_COARSE, FINAL_COARSE_KEY, FINAL_FINE, FINE_KEY


def train_recipes(names, data, output, device, folds=range(5), config_overrides=None):
    first, test, sample = load_dataset(data)
    second = []
    for item in first:
        other = copy.copy(item)
        other.fold = int(SPLIT2_FOLDS[Path(item.name).stem])
        second.append(other)
    root = output / 'runs'
    for name in names:
        config = copy.deepcopy(RECIPES[name])
        config.update((config_overrides or {}).get(name, {}))
        items = second if name in SECOND_SPLIT else first
        if 'coarse' in config:
            source, key = config.pop('coarse'), config.pop('coarse_key')
            run_fine(name, config, source, key, items, test, root, str(device), folds)
        else:
            run_coarse(name, config, items, test, root, str(device), folds)
    return first, test, sample


def export_cascade(root, train, test, sample, device, data):
    result = final_predictions(root / 'runs', train, test, device, FINAL_COARSE, FINAL_COARSE_KEY,
                               FINAL_FINE, FINE_KEY, iters=1, center_dec='p25')
    np.savez_compressed(root / 'cascade_predictions.npz', oof=result['oof'], test=result['test'],
                        train_names=np.asarray([item.name for item in train]),
                        test_names=np.asarray([item.name for item in test]),
                        target=np.asarray([item.xy for item in train]),
                        folds=np.asarray([item.fold for item in train]),
                        groups=np.asarray([item.group for item in train]), dataset_fingerprint=dataset_fingerprint(data))
    write_submission(test, result['test'], sample, root / 'submission_cascade.csv')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--out', type=Path, default=Path('outputs'))
    parser.add_argument('--group', choices=['final', 'experiments'], default='final')
    parser.add_argument('--run', choices=sorted(RECIPES))
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), default=list(range(5)))
    parser.add_argument('--device', default=str(default_device()))
    parser.add_argument('--list', action='store_true')
    args = parser.parse_args(argv)
    if args.list:
        print('\n'.join(RECIPES))
        return
    names = FINAL_COARSE + FINAL_FINE if args.group == 'final' else list(RECIPES)
    if args.run:
        names = [args.run]
        if 'coarse' in RECIPES[args.run]:
            names.insert(0, RECIPES[args.run]['coarse'])
    # Coarse outputs must exist before dependent fine runs.
    names = [name for name in names if 'coarse' not in RECIPES[name]] + [name for name in names if 'coarse' in RECIPES[name]]
    names = list(dict.fromkeys(names))
    args.out.mkdir(parents=True, exist_ok=True)
    train, test, sample = train_recipes(names, find_data_dir(args.data), args.out, torch.device(args.device), args.folds)
    ready = all((args.out / 'runs' / name / f'model_fold{fold}.pt').is_file()
                for name in FINAL_COARSE + FINAL_FINE for fold in range(5))
    if ready:
        export_cascade(args.out, train, test, sample, torch.device(args.device), find_data_dir(args.data))


if __name__ == '__main__':
    main()
