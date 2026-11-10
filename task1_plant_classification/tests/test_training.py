"""CPU checks of partial finetuning, numerical stability, and checkpoint recovery."""
import copy
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import numpy as np
import train_dinov2 as dino

class TrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        dino.load_torch()
        dino.torch.set_num_threads(1)

    def setUp(self):
        self.original = copy.deepcopy(dino.CONFIG)
        dino.CONFIG.update(epochs=3, head_only_epochs=1, workers=0, batch_size=2,
                          accumulation=2, unfreeze_blocks=2, backbone_lr=1e-3,
                          head_lr=1e-2)
        torch, nn = dino.torch, dino.nn

        class TinyBackbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embed_dim, self.num_register_tokens = 8, 0
                self.patch_embed = nn.Conv2d(3, 8, 2, 2)
                self.blocks = nn.ModuleList([nn.Sequential(nn.Linear(8,8), nn.GELU()) for _ in range(4)])
                self.norm = nn.LayerNorm(8)

            def prepare_tokens_with_masks(self, images):
                return self.patch_embed(images).flatten(2).transpose(1,2)

        self.factory = TinyBackbone
        dino.seed_everything(42)
        self.dataset = torch.utils.data.TensorDataset(torch.randn(8,3,8,8), torch.arange(8)%5)

    def tearDown(self):
        dino.CONFIG.clear()
        dino.CONFIG.update(self.original)

    def model(self):
        dino.seed_everything(123)
        return dino.make_model(self.factory())

    def test_frozen_prefix_stays_fixed_and_tail_learns(self):
        model = self.model()
        initial = {k:v.detach().clone() for k,v in model.named_parameters()}
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            dino.train_fold(model, self.dataset, dino.torch.device('cpu'),
                           Path(directory)/'state.pt', 'test', 0)
        for name, p in model.named_parameters():
            if not p.requires_grad:
                self.assertTrue(dino.torch.equal(initial[name], p), name)
        self.assertFalse(dino.torch.equal(initial['backbone.blocks.3.0.weight'],
                                        dict(model.named_parameters())['backbone.blocks.3.0.weight']))

    def test_head_warmup_has_no_backbone_gradients(self):
        model = self.model()
        model.train()
        model.head_only = True
        loss = dino.nn.functional.cross_entropy(model(self.dataset.tensors[0]), self.dataset.tensors[1])
        loss.backward()
        self.assertTrue(all(p.grad is None for p in model.backbone.parameters()))
        self.assertIsNotNone(model.head[-1].weight.grad)

    def test_fp16_scaled_gradient_overflows_while_fp32_remains_finite(self):
        torch = dino.torch
        model = dino.nn.Linear(2, 5)
        dino.nn.init.zeros_(model.weight)
        dino.nn.init.zeros_(model.bias)
        x, y = torch.full((2,2), 512.0), torch.zeros(2, dtype=torch.long)
        with torch.autocast('cpu', dtype=torch.float16):
            loss = dino.nn.functional.cross_entropy(model(x), y)
        self.assertTrue(torch.isfinite(loss))
        (loss*65536).backward()
        self.assertFalse(torch.isfinite(model.weight.grad).all())
        model.zero_grad(set_to_none=True)
        loss = dino.nn.functional.cross_entropy(model(x), y)
        loss.backward()
        norm = dino.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        self.assertTrue(torch.isfinite(norm))

    def test_true_nonfinite_loss_still_stops_without_checkpoint(self):
        model = self.model()
        model.head.register_forward_hook(lambda module, inputs, output: output*float('nan'))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            path = Path(directory)/'state.pt'
            with self.assertRaisesRegex(FloatingPointError, 'FP32 loss.*fold=1, epoch=1, batch=1'):
                dino.train_fold(model, self.dataset, dino.torch.device('cpu'), path, 'test', 0)
            self.assertFalse(path.exists())

    def test_completed_checkpoint_restores_same_ema_and_rejects_stale(self):
        torch = dino.torch
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            path = Path(directory)/'state.pt'
            model = self.model()
            dino.train_fold(model, self.dataset, torch.device('cpu'), path, 'test', 0)
            model.eval()
            expected = model(self.dataset.tensors[0]).detach()
            restored = self.model()
            dino.train_fold(restored, self.dataset, torch.device('cpu'), path, 'test', 0)
            restored.eval()
            torch.testing.assert_close(expected, restored(self.dataset.tensors[0]))
            with self.assertRaises(ValueError):
                dino.train_fold(restored, self.dataset, torch.device('cpu'), path, 'changed', 0)

    def test_image_views_predict_normalized_probabilities(self):
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(StringIO()):
            files = []
            for i in range(2):
                path = Path(directory)/f'{i}.png'
                Image.new('RGB', (25+i,30), (15,130,20)).save(path)
                files.append(path)
            dino.CONFIG['image_size'] = 28
            result = dino.predict(self.model(), files, dino.torch.device('cpu'))
            self.assertEqual(result.shape, (2,5))
            np.testing.assert_allclose(result.sum(1), 1, atol=1e-6)



if __name__ == '__main__':
    unittest.main()
