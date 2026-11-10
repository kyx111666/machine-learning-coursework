"""Synthetic coordinate geometry, heatmap decoding and submission checks."""
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from data import Fundus, apply, square, hflip, render, FROZEN_FOLDS, SPLIT2_FOLDS
from models import HeatNet, soft_argmax, heat_loss
from blend import FORMULAS, blend
from submission import write_submission
import classic


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_affine_round_trip_and_flip(self):
        matrix = square(100, 80, 640, 256)
        point = np.array([123.4, 72.1])
        np.testing.assert_allclose(apply(np.linalg.inv(matrix), apply(matrix, point)), point, atol=1e-10)
        np.testing.assert_allclose(hflip(256) @ hflip(256), np.eye(3))
        np.testing.assert_allclose(apply(matrix, [100, 80]), [127.5, 127.5])

    def test_duplicate_group_assignments(self):
        for assignment in [FROZEN_FOLDS, SPLIT2_FOLDS]:
            self.assertEqual(len(assignment), 80)
            self.assertEqual(assignment['0039'], assignment['0077'])
            self.assertEqual(set(map(int, assignment.values())), set(range(5)))

    def test_uniform_heatmap_decoding_and_loss(self):
        logits = torch.zeros(2, 8, 8, requires_grad=True)
        point = soft_argmax(logits, stride=4)
        torch.testing.assert_close(point, torch.full((2, 2), 15.5))
        heat_loss(logits, point.detach(), sigma=2.)[0].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())
        torch.testing.assert_close(classic.decode_heatmap(torch.zeros(2, 8, 8), 'sharp'), torch.full((2, 2), .5))

    def test_pyramid_render(self):
        item = Fundus('0001.jpg', np.full((96, 128, 3), 80, np.uint8), [64, 48], 0)
        view = render(item, square(64, 48, 80, 64), 64)
        self.assertEqual(view.shape, (64, 64, 3))
        self.assertEqual(view.dtype, np.uint8)

    def test_float64_coordinate_blending(self):
        a = np.asarray([[1234.5679, 987.6543]], np.float32)
        b = np.asarray([[1111.2222, 888.7778]], np.float32)
        result = blend({'a': a, 'b': b}, {'a': .7, 'b': .3})
        np.testing.assert_array_equal(result, .7*a.astype(np.float64) + .3*b.astype(np.float64))
        self.assertEqual(result.dtype, np.float64)
        for formula in FORMULAS.values(): self.assertAlmostEqual(sum(formula.values()), 1)
        with self.assertRaises(ValueError): blend({'a': a}, {'a': .4})
        with self.assertRaises(ValueError): blend({'a': np.full((1, 2), np.nan)}, {'a': 1})

    def test_submission_order_axes_and_bounds(self):
        items = [SimpleNamespace(name=f'{i:04d}.jpg', w=200, h=100) for i in range(81, 101)]
        points = np.asarray([[i, i/2] for i in range(20)], float)
        ids = [f'{i:04d}_Fovea_{axis}' for i in range(100, 80, -1) for axis in ['Y', 'X']]
        sample = pd.DataFrame({'ImageID': ids, 'value': 0})
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'submission.csv'
            saved = write_submission(items, points, sample, path)
            self.assertEqual(saved.ImageID.tolist(), ids)
            self.assertEqual(saved.value.iloc[:2].tolist(), [9.5, 19])
            bad = points.copy(); bad[0, 0] = 200
            with self.assertRaises(ValueError): write_submission(items, bad, sample, path)
            repeated = sample.copy(); repeated.iloc[0] = repeated.iloc[1]
            with self.assertRaises(ValueError): write_submission(items, points, repeated, path)

    def test_real_model_forward_backward(self):
        modern = HeatNet('resnet18', pretrained=False, dec=(32, 16, 8))
        image = torch.rand(2, 3, 64, 64)
        logits = modern(image)
        self.assertEqual(tuple(logits.shape), (2, 16, 16))
        heat_loss(logits, torch.full((2, 2), 31.5), 2.)[0].backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in modern.parameters() if p.grad is not None))
        for name in ['heatmap_dropout', 'resnet34_scratch']:
            xy, visible, heatmap = classic.Locator(4, name)(image)
            self.assertEqual(tuple(xy.shape), (2, 2))
            self.assertEqual(tuple(visible.shape), (2,))
            self.assertTrue(torch.isfinite(heatmap).all())


if __name__ == '__main__': unittest.main()
