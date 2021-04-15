import argparse
import json
import os
from model import Model
from alt_evaluator import AltEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')
parser.add_argument('-l', '--logdir', default='./logs/evaluate', help='directory to write logs')
parser.add_argument('-ld', '--lmdb', default='train.lmdb', help='The lmdb file name to be evaluated')
parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.pth')


def _eval(path_to_checkpoint_file, path_to_data_dir, path_to_log_dir, lmdb_file):
    path_to_eval_lmdb_dir = os.path.join(path_to_data_dir, lmdb_file)
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cpu()
    print(f'Evaluate {path_to_checkpoint_file} on {path_to_eval_lmdb_dir}')
    results = AltEvaluator(path_to_eval_lmdb_dir).evaluate(model)

    model_version = get_model_version(path_to_checkpoint_file)
    export_to_json(model_version, results, path_to_log_dir)


def get_model_version(path_to_checkpoint_file):
    start = path_to_checkpoint_file.find("-")+1
    end = path_to_checkpoint_file.find(".pth")
    substring = path_to_checkpoint_file[start:end]
    return substring 


def export_to_json(model_version, data, path_to_log_dir):
    with open(f'{path_to_log_dir}/data-{model_version}', 'w') as f:
        json.dump(data, f)


def main(args):
    path_to_data_dir = args.data_dir
    path_to_checkpoint_file = args.checkpoint
    log_dir = args.logdir
    lmdb_file = args.lmdb
    get_model_version(path_to_checkpoint_file)
    print('Start evaluate')
    _eval(path_to_checkpoint_file, path_to_data_dir, log_dir, lmdb_file)
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
