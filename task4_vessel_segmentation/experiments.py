"""Run the original 16-setting screening or the paired channel-wise Gamma experiment on Kaggle."""
import argparse
from pathlib import Path

import runtime as rt
from gamma import run_gamma_augmentation
from predictions import bundle_expanded_outputs
from train import require_kaggle, source_hash


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', default=None)
    parser.add_argument('--out', type=Path, default=Path('/kaggle/working/task4_experiments'))
    parser.add_argument('--group', choices=['screening', 'gamma'], default='screening')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args(argv)
    require_kaggle()
    if args.workers < 0:
        parser.error('--workers must be nonnegative')
    args.out.mkdir(parents=True, exist_ok=True)
    data = rt.OfficialData(rt.find_data(args.data), args.out/'data_checks')
    rt.verify_official_encoder(data.root, args.out/'data_checks')
    if args.group == 'screening':
        rt.run_analysis(data, args.out/'screening', source_hash(), workers=args.workers)
    else:
        directory = args.out/'gamma'
        try:
            run_gamma_augmentation(data, directory, source_hash(), args.workers)
        finally:
            if directory.exists():
                bundle_expanded_outputs(directory, args.out/'gamma_analysis.zip')


if __name__ == '__main__':
    main()
