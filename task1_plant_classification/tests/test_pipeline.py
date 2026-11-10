"""Synthetic checks of input integrity, probability blending, and submission order."""
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import data
import train_dinov2 as dino
import train_ensemble as ensemble


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.ids = np.asarray([f'test_{i:03d}.png' for i in range(378)])
        self.labels = np.repeat(np.arange(5), 100)
        self.template = pd.DataFrame({'ID': self.ids[::-1], 'Category': [''] * 378})
        self.rng = np.random.default_rng(7)

    def probabilities(self, n):
        return self.rng.dirichlet(np.ones(5), size=n).astype(np.float32)

    def test_shared_fold_balance_and_coverage(self):
        folds = data.fold_ids(self.labels)
        self.assertEqual(set(folds), set(range(5)))
        for fold in range(5):
            np.testing.assert_array_equal(np.bincount(self.labels[folds == fold]), [20] * 5)
        np.testing.assert_array_equal(folds, data.fold_ids(self.labels))

    def test_malformed_probabilities_rejected(self):
        for value in (np.full((378, 5), np.nan), np.ones((378, 5)),
                      np.full((377, 5), .2), np.tile([-.1, .2, .3, .3, .3], (378, 1))):
            with self.assertRaises(ValueError):
                data.validate_probabilities(value, 378, 'test')

    def test_submission_follows_template_and_rejects_bad_ids(self):
        probabilities = self.probabilities(378)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'submission.csv'
            data.write_submission(self.template, self.ids, probabilities, path)
            saved = pd.read_csv(path)
            self.assertEqual(saved.ID.tolist(), self.ids[::-1].tolist())
            expected = dict(zip(self.ids, data.CLASSES[probabilities.argmax(1)]))
            self.assertEqual(saved.Category.tolist(), [expected[key] for key in saved.ID])
            with self.assertRaises(ValueError):
                data.write_submission(self.template, np.repeat('bad', 378), probabilities, path)

    def test_both_blends_match_the_fixed_formula(self):
        conv_oof, conv_test = self.probabilities(500), self.probabilities(378)
        dino_oof, dino_test = self.probabilities(500), self.probabilities(378)
        bundle = dict(labels=self.labels, fold_id=data.fold_ids(self.labels),
                      convnext_oof=conv_oof, convnext_test=conv_test,
                      train_names=np.asarray([f'train_{i}' for i in range(500)]), test_names=self.ids)
        seeds = np.stack([self.probabilities(378) for _ in range(3)])
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            out = Path(directory)
            dino.write_results(bundle, self.template, dino_oof, dino_test, out)
            cv_test = ensemble.load_cv_probabilities(out / 'dinov2_cv_probabilities.npz', bundle)
            np.testing.assert_allclose(cv_test, .5 * conv_test + .5 * dino_test)
            result = ensemble.write_results(bundle, self.template, seeds, cv_test, out)
            np.testing.assert_allclose(result, .25 * conv_test + .25 * dino_test + .5 * seeds.mean(0), atol=1e-7)
            self.assertTrue((out / 'submission.csv').is_file())
            with self.assertRaises(ValueError):
                ensemble.write_results(bundle, self.template, seeds[:2], cv_test, out)
            wrong = dict(bundle, fold_id=np.roll(bundle['fold_id'], 1))
            with self.assertRaises(ValueError):
                ensemble.load_cv_probabilities(out / 'dinov2_cv_probabilities.npz', wrong)

    def test_only_one_artifact_source_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / 'output'
            output.mkdir()
            for name in ('first', 'second'):
                folder = root / name
                folder.mkdir()
                (folder / 'probabilities.npz').touch()
            with self.assertRaises(FileNotFoundError):
                data.find_artifact('probabilities.npz', output, root)
            path = output / 'probabilities.npz'
            path.touch()
            self.assertEqual(data.find_artifact('probabilities.npz', output, root), path)

    def test_input_bundle_created_from_images_without_legacy_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for cls in data.CLASSES:
                folder = root / 'train' / cls
                folder.mkdir(parents=True)
                for i in range(100):
                    Image.new('RGB', (4, 5), (i, 120, 30)).save(folder / f'{cls}_{i:03d}.png')
            (root / 'test').mkdir()
            for name in self.ids:
                Image.new('RGB', (4, 5), (10, 90, 30)).save(root / 'test' / name)
            self.template.to_csv(root / 'submission-for-task2.csv', index=False)
            train, test, _ = data.find_data(root)
            bundle = dict(train_names=np.asarray([p.name for p in train]), test_names=self.ids,
                          train_sha256=np.asarray([data.image_hash(p) for p in train]),
                          test_sha256=np.asarray([data.image_hash(p) for p in test]), labels=self.labels,
                          fold_id=data.fold_ids(self.labels), convnext_oof=self.probabilities(500),
                          convnext_test=self.probabilities(378))
            path = root / 'convnext_probabilities.npz'
            np.savez_compressed(path, **bundle)
            loaded, _, _, template = data.load_inputs(path, root)
            self.assertEqual(set(loaded), set(bundle))
            self.assertEqual(template.ID.tolist(), self.ids[::-1].tolist())
            Image.new('RGB', (4, 5), (220, 20, 30)).save(test[0])
            with self.assertRaisesRegex(ValueError, 'image content'):
                data.load_inputs(path, root)


if __name__ == '__main__':
    unittest.main()
