"""Shared command-line paths for the task 2 training entry points."""
import argparse
import json
import os


def configure(runs, default):
    parser = argparse.ArgumentParser(description='Task 2 emotion-classification experiments')
    parser.add_argument('--run', choices=runs, default=default)
    parser.add_argument('--input', default='/kaggle/input', help='Directory containing the 2026task2 input')
    parser.add_argument('--out', default='/kaggle/working/task2', help='Parent directory for run outputs')
    parser.add_argument('--data-root', help='Explicit competition image directory')
    parser.add_argument('--resume', help='Saved output directory to resume')
    parser.add_argument('--folds', nargs='+', type=int, choices=range(5), help='Optional subset of the shared five folds')
    parser.add_argument('--smoke', action='store_true', help='One-epoch workflow check on synthetic data')
    parser.add_argument('--list', action='store_true', help='List experiment groups and exit')
    args = parser.parse_args()
    if args.list:
        print('\n'.join(runs))
        parser.exit()
    os.environ['TASK2_RUN'] = args.run
    os.environ['TASK2_INPUT'] = args.input
    os.environ['TASK2_OUTPUT'] = args.out
    if args.folds:
        os.environ['TASK2_FOLDS'] = json.dumps(args.folds)
        os.environ['TASK2_SMOKE_FOLDS'] = str(len(args.folds))
    if args.smoke:
        os.environ['TASK2_SMOKE'] = '1'
    return args
