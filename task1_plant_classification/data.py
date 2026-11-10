"""Competition inputs, foreground crops, and prediction file validation."""
import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.model_selection import StratifiedKFold

CLASSES = np.asarray([
    "Black-grass", "Common wheat", "Loose Silky-bent",
    "Scentless Mayweed", "Sugar beet",
])
SEED = 2026
CROP_CONFIG = {
    "hue_range_pil": [35, 135], "min_saturation": 40,
    "green_red_ratio": 1.08, "green_blue_ratio": 1.10,
    "min_excess_green": 18, "min_component": 12,
    "component_largest_fraction": 0.015, "min_pixels": 50,
    "margin": 0.12, "min_side_fraction": 0.60,
}


def parse_paths(description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--data", type=Path, default=Path("/kaggle/input"))
    parser.add_argument("--out", type=Path, default=Path("/kaggle/working"))
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    return args


def competition_score(y, predictions):
    predicted = predictions.argmax(1) if predictions.ndim == 2 else predictions
    return float(np.mean(y == predicted))


def image_hash(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        return hashlib.sha256(
            np.asarray(image.size, dtype="<i8").tobytes() + image.tobytes()
        ).hexdigest()


def find_data(root):
    roots = [p.parent for p in root.rglob("train")
             if (p.parent / "test").is_dir()
             and [len(list((p / c).glob("*.png"))) for c in CLASSES] == [100] * 5]
    if len(roots) != 1:
        raise ValueError(f"Expected one competition directory with 100 images per class; found {len(roots)}")
    train = [p for c in CLASSES for p in sorted((roots[0] / "train" / c).glob("*.png"))]
    test = sorted((roots[0] / "test").glob("*.png"))
    if len(test) != 378 or len({p.name for p in train}) != 500:
        raise ValueError("Expected 500 distinct training filenames and 378 test images")
    template_path = roots[0] / "submission-for-task2.csv"
    if not template_path.is_file():
        raise FileNotFoundError(f"Official template missing: {template_path}")
    template = pd.read_csv(template_path, dtype={"ID": str})
    if (set(template.columns) != {"ID", "Category"} or len(template) != 378
            or not template.ID.is_unique or set(template.ID) != {p.name for p in test}):
        raise ValueError("Official template IDs do not match test images")
    return train, test, template[["ID", "Category"]]


def fold_ids(labels):
    result = np.empty(len(labels), dtype=np.int64)
    splitter = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for fold, (_, valid) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        result[valid] = fold
    return result


def validate_probabilities(array, n, name):
    if (array.shape != (n, 5) or not np.isfinite(array).all()
            or array.min() < 0 or not np.allclose(array.sum(1), 1, atol=1e-4)):
        raise ValueError(f"Invalid probability matrix: {name}")


def load_inputs(bundle_path, search_root):
    with np.load(bundle_path, allow_pickle=False) as data:
        bundle = {key: data[key].copy() for key in data.files}
    train, test, template = find_data(search_root)
    for split, files in (("train", train), ("test", test)):
        if not np.array_equal(bundle[f"{split}_names"], [p.name for p in files]):
            raise ValueError(f"{split} filenames or order differ from ConvNeXt output")
        if not np.array_equal(bundle[f"{split}_sha256"], [image_hash(p) for p in files]):
            raise ValueError(f"{split} image content differs from ConvNeXt output")
    labels = np.repeat(np.arange(5), 100)
    if not np.array_equal(bundle["labels"], labels):
        raise ValueError("Training label order differs from the competition data")
    if not np.array_equal(bundle["fold_id"], fold_ids(labels)):
        raise ValueError("Fold assignment differs from the shared five-fold split")
    validate_probabilities(bundle["convnext_oof"], 500, "ConvNeXt OOF")
    validate_probabilities(bundle["convnext_test"], 378, "ConvNeXt test")
    return bundle, train, test, template


def find_artifact(name, output, data_root):
    local = output / name
    if local.is_file():
        return local
    candidates = sorted(data_root.rglob(name))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected one {name} in --out or --data; found {len(candidates)}")
    return candidates[0]


def write_submission(template, names, probabilities, path):
    validate_probabilities(probabilities, 378, path.name)
    if (len(names) != 378 or len(set(names)) != 378 or len(template) != 378
            or set(template.ID.astype(str)) != set(names)):
        raise ValueError("Submission IDs do not match prediction IDs")
    frame = template.copy()
    mapping = dict(zip(names, CLASSES[probabilities.argmax(1)]))
    frame["Category"] = frame.ID.astype(str).map(mapping)
    if frame.Category.isna().any() or not frame.ID.is_unique:
        raise ValueError("Submission contains missing classes or duplicate IDs")
    frame.to_csv(path, index=False)


def foreground_box(image):
    """仅依据本图颜色裁剪；每条边至少保留原图的 60%。"""
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    hsv = np.asarray(image.convert("HSV"))
    r, g, b = rgb.transpose(2, 0, 1)
    lo, hi = CROP_CONFIG["hue_range_pil"]
    mask = (
        (hsv[:, :, 0] >= lo) & (hsv[:, :, 0] <= hi)
        & (hsv[:, :, 1] >= CROP_CONFIG["min_saturation"])
        & (g > CROP_CONFIG["green_red_ratio"] * r)
        & (g > CROP_CONFIG["green_blue_ratio"] * b)
        & (2 * g - r - b > CROP_CONFIG["min_excess_green"])
    )
    # 保留边缘植物，避免形态学操作把贴边细叶删掉。
    mask = mask | ndimage.binary_closing(mask, iterations=2)
    components, _ = ndimage.label(mask)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    if sizes.max() > 0:
        minimum = max(CROP_CONFIG["min_component"],
                      CROP_CONFIG["component_largest_fraction"] * sizes.max())
        mask = sizes[components] >= minimum
    yy, xx = np.where(mask)
    if len(yy) < CROP_CONFIG["min_pixels"]:
        return (0, 0, image.width, image.height), float(mask.mean())
    margin = max(4, int(np.ceil(CROP_CONFIG["margin"] * max(np.ptp(xx) + 1, np.ptp(yy) + 1))))
    bounds = []
    for points, length in ((xx, image.width), (yy, image.height)):
        start, end = int(points.min()), int(points.max()) + 1
        span = min(length, max(end - start + 2 * margin,
                               int(np.ceil(CROP_CONFIG["min_side_fraction"] * length))))
        start = min(max(0, int(np.floor((start + end - span) / 2))), length - span)
        bounds.append((start, start + span))
    return (bounds[0][0], bounds[1][0], bounds[0][1], bounds[1][1]), float(mask.mean())
