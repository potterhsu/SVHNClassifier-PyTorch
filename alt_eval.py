import argparse
import json
import os
from model import Model
from alt_evaluator import AltEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')
parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.pth')


def _eval(path_to_checkpoint_file, path_to_eval_lmdb_dir):
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cpu()
    results = AltEvaluator(path_to_eval_lmdb_dir).evaluate(model)
    print(f'Evaluate {path_to_checkpoint_file} on {path_to_eval_lmdb_dir}, results =')
    print(results)


def main(args):
    path_to_test_lmdb_dir = os.path.join(args.data_dir, 'test.lmdb')
    path_to_checkpoint_file = args.checkpoint
    print('Start evaluate')
    _eval(path_to_checkpoint_file, path_to_test_lmdb_dir)
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
