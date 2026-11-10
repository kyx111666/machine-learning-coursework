"""Official 20-image, 40-row coordinate submission format."""
import re
import numpy as np
from data import stem_key


def write_submission(test, pred, sample, path):
    pred = np.asarray(pred)
    if len(test) != 20 or pred.shape != (20, 2) or not np.isfinite(pred).all():
        raise ValueError('Expected twenty finite (x, y) predictions')
    mapping = {stem_key(fd.name): point for fd, point in zip(test, pred)}
    if len(mapping) != 20 or list(sample) != ['ImageID', 'value'] or len(sample) != 40:
        raise ValueError('Invalid official image names or submission template')
    for fd, point in zip(test, pred):
        if not (0 <= point[0] < fd.w and 0 <= point[1] < fd.h):
            raise ValueError(f'Coordinate outside the original image: {fd.name}')
    values, seen = [], set()
    for image_id in sample.ImageID:
        match = re.fullmatch(r'(.+)_Fovea_([XY])', str(image_id))
        if match is None:
            raise ValueError('Invalid coordinate ID in submission template')
        name, axis = stem_key(match[1]), match[2]
        if name not in mapping or (name, axis) in seen:
            raise ValueError('Unknown or repeated coordinate ID')
        seen.add((name, axis))
        values.append(float(mapping[name][0 if axis == 'X' else 1]))
    if len(seen) != 40:
        raise ValueError('Incomplete coordinate submission')
    frame = sample.copy()
    frame['value'] = values
    frame.to_csv(path, index=False)
    return frame
