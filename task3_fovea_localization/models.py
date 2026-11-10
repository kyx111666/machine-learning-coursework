"""Heatmap U-Net on an ImageNet-pretrained timm encoder, soft-argmax decoding and the training loss."""
import torch
from torch import nn
from torch.nn import functional as F

class ConvGN(nn.Sequential):
    def __init__(self, a, b):
        super().__init__(nn.Conv2d(a, b, 3, padding=1, bias=False), nn.GroupNorm(8, b), nn.ReLU(inplace=True))

class HeatNet(nn.Module):
    """Encoder features at strides 4-32 are fused top-down; one logit map at stride 4 is returned."""
    stride = 4

    def __init__(self, encoder='resnet34', pretrained=True, dec=(256, 128, 64), drop=0.):
        super().__init__()
        import timm
        self.enc = timm.create_model(encoder, pretrained=pretrained, features_only=True)
        red, chs = self.enc.feature_info.reduction(), self.enc.feature_info.channels()
        self.pick = [red.index(r) for r in (4, 8, 16, 32)]
        c = [chs[i] for i in self.pick]
        self.d3 = nn.Sequential(ConvGN(c[3] + c[2], dec[0]), ConvGN(dec[0], dec[0]))
        self.d2 = nn.Sequential(ConvGN(dec[0] + c[1], dec[1]), ConvGN(dec[1], dec[1]))
        self.d1 = nn.Sequential(ConvGN(dec[1] + c[0], dec[2]), ConvGN(dec[2], dec[2]))
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.head = nn.Conv2d(dec[2], 1, 1)
        cfg = getattr(self.enc, 'pretrained_cfg', {}) or {}
        self.register_buffer('mean', torch.tensor(cfg.get('mean', (.485, .456, .406))).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor(cfg.get('std', (.229, .224, .225))).view(1, 3, 1, 1))

    def forward(self, x):
        """x: RGB in [0, 1], shape (B, 3, S, S); returns logits (B, S/4, S/4)."""
        f = self.enc((x - self.mean) / self.std)
        f1, f2, f3, f4 = (f[i] for i in self.pick)

        def up(a, b):
            return torch.cat([F.interpolate(a, size=b.shape[-2:], mode='bilinear', align_corners=False), b], 1)

        y = self.d3(up(f4, f3))
        y = self.d2(up(y, f2))
        y = self.d1(up(y, f1))
        return self.head(self.drop(y)).squeeze(1)

def grid(n, stride, device):
    """Input-pixel coordinate of each heatmap cell centre (OpenCV pixel-centre convention)."""
    return (torch.arange(n, device=device, dtype=torch.float32) + .5) * stride - .5

def soft_argmax(logits, stride=4, temperature=1., window=0, prior_sigma=0.):
    """Expected position under softmax(logits / T).

    window keeps only cells near the peak; prior_sigma (input pixels) adds a Gaussian log-prior centred on the
    input centre, i.e. on the coarse estimate that the fine crop was centred on."""
    B, H, W = logits.shape
    z = logits.float().flatten(1) / temperature
    if prior_sigma:
        c = (W * stride - 1) / 2
        d2 = (grid(W, stride, z.device).view(1, W) - c) ** 2 + (grid(H, stride, z.device).view(H, 1) - c) ** 2
        z = z - (d2 / (2 * prior_sigma ** 2)).flatten()[None]
    if window:
        peak = z.argmax(1)
        py, px = (peak // W).view(-1, 1, 1), (peak % W).view(-1, 1, 1)
        yy = torch.arange(H, device=z.device).view(1, H, 1)
        xx = torch.arange(W, device=z.device).view(1, 1, W)
        keep = ((yy - py).abs() <= window) & ((xx - px).abs() <= window)
        z = z.masked_fill(~keep.flatten(1), float('-inf'))
    p = z.softmax(1).view(B, H, W)
    return torch.stack([(p.sum(1) * grid(W, stride, z.device)).sum(1),
                        (p.sum(2) * grid(H, stride, z.device)).sum(1)], 1)

DECODERS = {'global': dict(), 'sharp': dict(temperature=.5), 'local': dict(window=6)}

def heat_loss(logits, xy, sigma, coord_w=1., stride=4):
    """Cross-entropy to a normalised Gaussian target plus L1 of the soft-argmax (in heatmap cells)."""
    B, H, W = logits.shape
    gx, gy = grid(W, stride, logits.device), grid(H, stride, logits.device)
    d2 = (gx.view(1, 1, W) - xy[:, 0].view(-1, 1, 1)) ** 2 + (gy.view(1, H, 1) - xy[:, 1].view(-1, 1, 1)) ** 2
    t = torch.exp(-d2 / (2 * sigma ** 2)).flatten(1)
    t = t / t.sum(1, keepdim=True).clamp_min(1e-12)
    ce = -(t * logits.float().flatten(1).log_softmax(1)).sum(1).mean()
    l1 = (soft_argmax(logits, stride) - xy).abs().sum(1).mean() / stride
    return ce + coord_w * l1, ce.detach(), l1.detach()
