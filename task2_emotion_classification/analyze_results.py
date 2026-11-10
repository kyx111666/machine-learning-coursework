"""Summarize model selection, paired ablations, hyperparameters, EMA and TTA from training outputs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

from blend import cv_weights, load_runs, weighted_mean

LABELS = ['Angry', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']


def paired_interval(correct, reference, groups, draws=5000):
    _, inverse = np.unique(groups, return_inverse=True)
    counts = np.bincount(inverse)
    sums = np.bincount(inverse, weights=correct.astype(float) - reference.astype(float))
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(draws):
        selected = rng.integers(0, len(counts), len(counts))
        samples.append(100 * sums[selected].sum() / counts[selected].sum())
    low, high = np.quantile(samples, [.025, .975])
    return 100 * float(correct.mean() - reference.mean()), float(low), float(high)


def summarize_runs(root):
    models, tta, ablations, hyperparameters = [], [], [], []
    curves = []
    for plan_path in sorted(root.rglob('plan.json')):
        directory = plan_path.parent
        if not (directory / 'summary.csv').is_file():
            continue
        plan = json.loads(plan_path.read_text())
        status = json.loads((directory / 'status.json').read_text())
        index = pd.read_csv(directory / 'train_index.csv')
        configs = {c['name']: c for c in plan}
        reference = None
        for config in plan:
            name = config['name']
            path = directory / f'oof_{name}.npz'
            if not path.exists():
                continue
            with np.load(path, allow_pickle=False) as saved:
                mask, labels = saved['mask'].copy(), saved['labels'].copy()
                predictions = {k: saved[k].copy()[mask] for k in ['ema_none', 'ema_flip', 'ema_tta10', 'raw_flip']}
            if not mask.any() or not np.array_equal(labels, index.label):
                raise ValueError(f'{directory}: incomplete or mismatched OOF labels')
            true = labels[mask]
            scores = {key: float((prob.argmax(1) == true).mean()) for key, prob in predictions.items()}
            run = str(directory.relative_to(root))
            models.append({'run': run, 'experiment': name, 'model': config['arch'],
                           'pretrained': config['pretrained'], 'image_size': config['img_size'],
                           'epochs': config['epochs'], 'learning_rate': config['lr'],
                           'seed': config['seed'], 'evaluated_rows': int(mask.sum()),
                           'full_oof': bool(mask.all()), 'split': status['split_fingerprint'],
                           'accuracy': scores['ema_tta10'],
                           'macro_f1': float(f1_score(true, predictions['ema_tta10'].argmax(1), average='macro', labels=range(6), zero_division=0))})
            tta.append({'run': run, 'experiment': name, **scores})
            if reference is None:
                reference = (mask, true, predictions['ema_flip'].argmax(1) == true)
            if name.startswith(('A', 'F')):
                if not np.array_equal(mask, reference[0]):
                    raise ValueError('Ablations do not cover the same validation rows')
                correct = predictions['ema_flip'].argmax(1) == true
                delta, low, high = paired_interval(correct, reference[2], index.group.to_numpy()[mask])
                regime = 'initial' if name.startswith('A') else 'final'
                ablations.append({'regime': regime, 'experiment': name, 'accuracy': scores['ema_flip'],
                                  'delta_pp': delta, 'ci95_low_pp': low, 'ci95_high_pp': high,
                                  'corrected': int((correct & ~reference[2]).sum()),
                                  'harmed': int((~correct & reference[2]).sum()), 'evaluated_rows': len(true)})
                base = plan[0]
                for parameter in ['lr', 'img_size', 'epochs', 'wd']:
                    ignored = {'name', parameter}
                    if all(config.get(k) == base.get(k) for k in set(config) | set(base) if k not in ignored):
                        hyperparameters.append({'regime': regime, 'parameter': parameter,
                                                'value': config[parameter], 'accuracy': scores['ema_flip'],
                                                'delta_pp': delta, 'ci95_low_pp': low, 'ci95_high_pp': high})
            for fold in config['folds']:
                history = directory / 'experiments' / name / f'fold{fold}' / 'history.csv'
                if history.is_file():
                    curves.append((run, name, fold, pd.read_csv(history)))
    return pd.DataFrame(models), pd.DataFrame(tta), pd.DataFrame(ablations), pd.DataFrame(hyperparameters), curves


def plot_tables(models, ablations, hyperparameters, curves, output):
    full = models[models.full_oof] if len(models) else models
    if len(full):
        fig, ax = plt.subplots(figsize=(10, max(4, len(full) * .35)))
        ax.barh(full.run + '/' + full.experiment, full.accuracy, color='#527a9b')
        ax.set_xlim(0, 1)
        ax.set_xlabel('OOF accuracy (EMA + 10-view TTA)')
        fig.tight_layout(); fig.savefig(output / 'model_comparison.png', dpi=160); plt.close(fig)
    if len(ablations):
        for regime, frame in ablations.groupby('regime'):
            fig, ax = plt.subplots(figsize=(10, max(4, len(frame) * .35)))
            points = np.arange(len(frame))
            # Endpoints remain correct even when a percentile interval excludes its point estimate.
            ax.hlines(points, frame.ci95_low_pp, frame.ci95_high_pp, color='#527a9b')
            ax.scatter(frame.delta_pp, points, color='#244b6b', s=20)
            ax.set_yticks(points, frame.experiment)
            ax.axvline(0, color='black', linewidth=.7)
            ax.set_xlabel('Accuracy change (percentage points; paired group bootstrap 95% CI)')
            ax.invert_yaxis(); fig.tight_layout()
            fig.savefig(output / f'ablation_{regime}.png', dpi=160); plt.close(fig)
    for (regime, parameter), frame in hyperparameters.groupby(['regime', 'parameter']) if len(hyperparameters) else []:
        frame = frame.sort_values('value')
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(frame.value, frame.accuracy, marker='o', color='#244b6b')
        if parameter in ['lr', 'wd']:
            ax.set_xscale('log')
        ax.set_xlabel(parameter); ax.set_ylabel('Validation accuracy (EMA + flip)')
        ax.set_title(f'{regime} recipe'); fig.tight_layout()
        fig.savefig(output / f'hyperparameter_{regime}_{parameter}.png', dpi=160); plt.close(fig)
    for run, name, fold, history in curves:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(history.epoch, history.train_loss, color='#244b6b')
        axes[0].set_ylabel('Training loss')
        axes[1].plot(history.epoch, history.val_acc_ema, label='EMA')
        axes[1].plot(history.epoch, history.val_acc_raw, label='Raw weights')
        axes[1].set_ylabel('Validation accuracy'); axes[1].legend()
        for ax in axes: ax.set_xlabel('Epoch')
        fig.suptitle(f'{name}, fold {fold}'); fig.tight_layout()
        filename = f'{run}_{name}_fold{fold}'.replace('/', '_')
        fig.savefig(output / f'curve_{filename}.png', dpi=140); plt.close(fig)


def ensemble_analysis(root, recipe, output):
    index, _, oof, _ = load_runs(root, recipe)
    probabilities = weighted_mean(oof, cv_weights(recipe))
    labels = index.label.to_numpy()
    predicted = probabilities.argmax(1)
    matrix = confusion_matrix(labels, predicted, labels=range(6))
    pd.DataFrame(matrix, index=LABELS, columns=LABELS).to_csv(output / f'{recipe}_confusion.csv')
    normalized = matrix / np.maximum(matrix.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(normalized, cmap='Blues', vmin=0, vmax=1)
    for i in range(6):
        for j in range(6):
            ax.text(j, i, f'{normalized[i, j]:.2f}', ha='center', va='center',
                    color='white' if normalized[i, j] > .5 else 'black')
    ax.set_xticks(range(6), LABELS, rotation=30); ax.set_yticks(range(6), LABELS)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    fig.colorbar(im); fig.tight_layout(); fig.savefig(output / f'{recipe}_confusion.png', dpi=160); plt.close(fig)
    metrics = {'recipe': recipe, 'oof_accuracy_without_holdout_model': float((predicted == labels).mean()),
               'macro_f1': float(f1_score(labels, predicted, average='macro'))}
    (output / f'{recipe}_metrics.json').write_text(json.dumps(metrics, indent=2))
    comparisons = []
    reference = predicted == labels
    for name, probability in oof.items():
        delta, low, high = paired_interval(reference, probability.argmax(1) == labels, index.group.to_numpy())
        comparisons.append({'reference': name, 'ensemble_gain_pp': delta, 'ci95_low_pp': low, 'ci95_high_pp': high})
    pd.DataFrame(comparisons).to_csv(output / f'{recipe}_model_gains.csv', index=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--ensemble', choices=['seeds', 'vgg'], help='Also evaluate a complete final CV ensemble')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    models, tta, ablations, hyperparameters, curves = summarize_runs(args.results)
    if models.empty:
        raise ValueError('No completed OOF results found')
    for name, table in [('model_comparison', models), ('tta_ema', tta), ('ablation_comparison', ablations), ('hyperparameters', hyperparameters)]:
        if len(table): table.to_csv(args.out / f'{name}.csv', index=False)
    plot_tables(models, ablations, hyperparameters, curves, args.out)
    if args.ensemble:
        ensemble_analysis(args.results, args.ensemble, args.out)
    print(f'Tables and plots saved to {args.out}')


if __name__ == '__main__':
    main()
