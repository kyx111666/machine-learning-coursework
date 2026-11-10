"""Non-training checks for augmentation, fixed fusion, masks and submission encoding."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from PIL import Image
import torch

import runtime as rt
from gamma import GammaPatchDataset, gamma_chw, gamma_gate, gamma_recipes
from train import final_recipes, fuse, require_kaggle, run_final


class PipelineTests(unittest.TestCase):
    def setUp(self):
        # These tests must never construct a model or start an optimizer.
        for target in ['runtime.VesselUNet', 'torch.optim.AdamW']:
            guard = patch(target, side_effect=AssertionError('Training is prohibited in these checks'))
            guard.start()
            self.addCleanup(guard.stop)

    def test_rle_round_trip_and_invalid_runs(self):
        rng = np.random.default_rng(7)
        for mask in [np.zeros((512, 512), np.uint8), np.ones((512, 512), np.uint8),
                     (rng.random((512, 512)) < .03).astype(np.uint8), np.eye(512, dtype=np.uint8)]:
            np.testing.assert_array_equal(rt.rle_decode(rt.rle_encode(mask)), mask)
        self.assertEqual(rt.rle_encode(np.array([[1, 1, 0], [0, 1, 1]])), '1 2 5 2')
        for value in ['0 1', '1 -1', '1 4 3 2', '262144 2', '1']:
            with self.assertRaises(AssertionError):
                rt.rle_decode(value)

    def test_native_threshold_then_official_resize(self):
        prob = np.zeros((584, 565), np.float32)
        prob[100:120, 80:90] = .425
        prob[300, 300] = .9
        expected = np.asarray(Image.fromarray((prob >= .425).astype(np.uint8)*255)
                              .resize((512, 512), Image.Resampling.BILINEAR)) > 0
        np.testing.assert_array_equal(rt.output_mask(prob, .425, 0, 'native_then_official'), expected)
        filtered = rt.output_mask(prob, .425, 8, 'native_then_official')
        self.assertLess(filtered.sum(), expected.sum())
        self.assertGreater(filtered.sum(), 8)
        self.assertEqual(int(rt.component_filter(np.eye(3), 3).sum()), 3)

    def test_fixed_fusion_preserves_float32_arithmetic(self):
        rng = np.random.default_rng(2026)
        parts = {key: rng.random((2, 17, 13), dtype=np.float32) for key in ['rgb1', 'rgb2', 'green']}
        expected = .75*(.5*parts['rgb1']+.5*parts['rgb2'])+.25*parts['green']
        np.testing.assert_array_equal(fuse(parts, 'seed_blend'), expected)
        expected = .375*parts['rgb1']+.375*parts['rgb2']+.25*parts['green']
        np.testing.assert_array_equal(fuse(parts, 'gamma'), expected)
        with self.assertRaises(ValueError):
            fuse({**parts, 'rgb2': np.full((2, 17, 13), np.nan)}, 'gamma')
        with self.assertRaises(ValueError):
            fuse({**parts, 'rgb2': np.zeros((2, 16, 13))}, 'gamma')
        with self.assertRaises(ValueError):
            fuse(parts, 'unknown')

    def test_final_recipes_and_screening(self):
        for route in ['seed_blend', 'gamma']:
            recipes = final_recipes(route)
            self.assertEqual([recipes[k].seed for k in ['rgb1', 'rgb2', 'green']],
                             [20261031, 20261102, 20261031])
            self.assertEqual(recipes['green'].arch, 'unet')
            self.assertEqual(recipes['rgb1'].arch, 'resunet')
            for cfg in recipes.values():
                self.assertEqual((cfg.patch, cfg.width, cfg.lr, cfg.dice_weight), (192, 24, 8e-4, .5))
        screening = rt.screening_recipes()
        self.assertEqual(len(screening), 16)
        self.assertEqual(len({cfg.name for cfg in screening}), 16)

    def test_gamma_keeps_crop_geometry_and_labels(self):
        rng = np.random.default_rng(9)
        image = rng.integers(0, 256, (584, 565, 3), dtype=np.uint8)
        label = (image[..., 0] > 127).astype(np.uint8)
        recipes = gamma_recipes()
        before = rt.PatchDataset([image], [label], [0], recipes['reference'], 24, 20261102)
        after = GammaPatchDataset([image], [label], [0], recipes['gamma'], 24, 20261102)
        changed = 0
        for index in range(24):
            x, y = before[index]
            augmented, target = after[index]
            self.assertTrue(torch.equal(y, target))
            np.testing.assert_array_equal(augmented.numpy(), gamma_chw(x.numpy(), after.seed, index))
            self.assertTrue(torch.isfinite(augmented).all())
            self.assertTrue(augmented.min() >= -1 and augmented.max() <= 1)
            changed += int(not torch.equal(x, augmented))
        self.assertGreater(changed, 0)
        self.assertLess(changed, 24)

    def test_gamma_gate_requires_confirmation_folds(self):
        frame = pd.DataFrame({'fold': range(5), 'reference': .85, 'gamma_candidate': .851,
                              'gain': [.002, -.0001, .002, -.0001, .002]})
        self.assertTrue(gamma_gate(frame[frame.fold.isin([0, 2, 4])], 'fusion_pilot')['accepted'])
        self.assertFalse(gamma_gate(frame, 'confirmation')['accepted'])
        frame.loc[frame.fold == 1, 'gain'] = .0003
        self.assertTrue(gamma_gate(frame, 'confirmation')['accepted'])
        with self.assertRaises(AssertionError):
            gamma_gate(frame.iloc[:3], 'fusion_pilot')

    def test_submission_rejects_invalid_probability_maps(self):
        with tempfile.TemporaryDirectory() as tmp:
            for probs in [np.zeros((19, 584, 565), np.float32),
                          np.full((20, 584, 565), np.nan, np.float32),
                          np.full((20, 584, 565), 1.01, np.float32)]:
                with self.assertRaises(ValueError):
                    rt.make_submission(SimpleNamespace(), probs, {}, Path(tmp)/'selected')
            self.assertFalse((Path(tmp)/'selected').exists())

    def test_training_entry_rejects_local_cpu(self):
        with patch.object(rt, 'DEVICE', SimpleNamespace(type='cpu')):
            with self.assertRaises(RuntimeError):
                require_kaggle()

    def test_final_runner_rejects_changed_run_identity(self):
        with tempfile.TemporaryDirectory() as tmp, patch('train.require_kaggle'):
            directory = Path(tmp)
            (directory/'plan.json').write_text('{}')
            with self.assertRaises(ValueError):
                run_final(SimpleNamespace(fingerprint='changed-input'), directory, 'gamma', 'new-source')


if __name__ == '__main__':
    unittest.main()
