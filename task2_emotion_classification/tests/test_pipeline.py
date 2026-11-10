"""Synthetic probability-file, split-integrity and model-interface checks."""
import ast
import copy
import json
import tempfile
import unittest
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import timm

import blend
from analyze_results import paired_interval

ROOT = Path(__file__).resolve().parents[1]


def definitions(filename, names):
    tree = ast.parse((ROOT / filename).read_text())
    namespace = {'torch': torch, 'nn': nn, 'F': F, 'timm': timm, 'copy': copy,
                 'defaultdict': defaultdict}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), filename, 'exec'), namespace)
    return namespace


class PipelineTests(unittest.TestCase):
    def test_weights_and_family_allocation(self):
        for recipe, weights in blend.WEIGHTS.items():
            self.assertAlmostEqual(sum(weights.values()), 1)
            self.assertAlmostEqual(sum(blend.cv_weights(recipe).values()), 1)
        self.assertEqual(blend.WEIGHTS['seeds']['holdout_resnet32'], .25)
        self.assertEqual(blend.WEIGHTS['seeds']['resnet32_seed2027'], .125)
        self.assertEqual(blend.WEIGHTS['vgg']['vgg'], .25)
        self.assertEqual(blend.WEIGHTS['vgg']['holdout_resnet32'], .125)

    def test_blend_rejects_invalid_inputs(self):
        good = np.full((4, 6), 1 / 6)
        np.testing.assert_allclose(blend.weighted_mean({'a': good, 'b': good}, {'a': .5, 'b': .5}), good)
        for bad in [np.ones((4, 6)), np.full((4, 6), np.nan), np.full((3, 6), 1 / 6)]:
            with self.assertRaises(ValueError):
                blend.weighted_mean({'a': good, 'b': bad}, {'a': .5, 'b': .5})
        with self.assertRaises(ValueError):
            blend.weighted_mean({'a': good}, {'a': .4})

    def test_csv_preserves_ids(self):
        ids = np.asarray(['image_b', '0001', 'image_a'])
        probabilities = np.eye(6)[[2, 5, 0]]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'submission.csv'
            blend.write_submission(ids, probabilities, path)
            saved = pd.read_csv(path, dtype={'ID': str})
            self.assertEqual(saved.ID.tolist(), ids.tolist())
            self.assertEqual(saved.Emotion.tolist(), [2, 5, 0])

    def test_five_fold_mean_and_group_integrity(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            experiment = 'model'
            ids = np.asarray(['b.png', 'a.png'])
            index = pd.DataFrame({'relative_path': [f'image{i}' for i in range(30)],
                                  'label': np.arange(30) % 6, 'fold': np.arange(30) % 5,
                                  'group': [f'group{i}' for i in range(30)]})
            index.to_csv(root / 'train_index.csv', index=False)
            status = {'pending': [], 'folds_done': {experiment: list(range(5))}}
            (root / 'status.json').write_text(json.dumps(status))
            probabilities = np.full((2, 6), 1 / 6, dtype=np.float32)
            np.savez_compressed(root / 'oof_model.npz', labels=index.label, mask=np.ones(30, bool),
                                ema_tta10=np.full((30, 6), 1 / 6))
            np.savez_compressed(root / 'test_model.npz', ids=ids, ema_tta10=probabilities)
            for fold in range(5):
                folder = root / 'experiments' / experiment / f'fold{fold}'
                folder.mkdir(parents=True)
                np.savez_compressed(folder / 'test_probs.npz', ids=ids, ema_tta10=probabilities)
            blend.read_cv_run(root, experiment)
            np.savez_compressed(root / 'experiments/model/fold0/test_probs.npz', ids=ids[::-1], ema_tta10=probabilities)
            with self.assertRaisesRegex(ValueError, 'test IDs differ'):
                blend.read_cv_run(root, experiment)
            index.loc[1, 'group'] = index.loc[0, 'group']
            index.to_csv(root / 'train_index.csv', index=False)
            with self.assertRaisesRegex(ValueError, 'group crosses'):
                blend.read_cv_run(root, experiment)

    def test_group_bootstrap_identity(self):
        correct = np.array([True, False, True, False])
        self.assertEqual(paired_interval(correct, correct, np.array([0, 0, 1, 1]), draws=20), (0, 0, 0))

    def test_local_scratch_model_and_ema_forward(self):
        torch.set_num_threads(1)
        ns = definitions('train_resnet32.py', {'conv_bn', 'ResidualBlock', 'SmallResNet', 'EMA'})
        model = ns['SmallResNet'](5).eval()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 1859430)
        with torch.no_grad():
            output = model(torch.randn(2, 1, 48, 48))
        self.assertEqual(tuple(output.shape), (2, 6))
        ema = ns['EMA'](model, .999)
        before = copy.deepcopy(ema.module.state_dict())
        with torch.no_grad():
            next(model.parameters()).add_(1)
        ema.update(model, 0)
        name = next(iter(dict(model.named_parameters())))
        self.assertFalse(torch.equal(before[name], ema.module.state_dict()[name]))

    def test_final_pretrained_and_vgg_interfaces(self):
        torch.set_num_threads(1)
        ns = definitions('train_pretrained.py', {'BasicBlock', 'CifarResNet18', 'VGGNet', '_create', 'build_model'})
        for arch in ['resnet18.a1_in1k', 'efficientnet_b0.ra_in1k', 'deit_tiny_patch16_224.fb_in1k',
                     'vit_small_patch16_224.augreg_in1k', 'convnext_tiny.fb_in1k', 'vgg_scratch']:
            with self.subTest(arch=arch):
                config = dict(arch=arch, pretrained=False, img_size=112, drop_path=0.0)
                model = ns['build_model'](config).eval()
                size = 48 if arch == 'vgg_scratch' else 112
                with torch.no_grad(): output = model(torch.randn(2, 1, size, size))
                self.assertEqual(tuple(output.shape), (2, 6))
                del model


if __name__ == '__main__':
    unittest.main()
