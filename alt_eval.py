import argparse
import json
import os

from .model import Model
from pathlib import Path
from .alt_evaluator import AltEvaluator

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')
parser.add_argument('-l', '--logdir', default='./logs/evaluate', help='directory to write logs')
parser.add_argument('-ld', '--lmdb', default='train.lmdb', help='The lmdb file name to be evaluated')
parser.add_argument('-n', '--num_images', default=None, help='The number of images to be evaluated')
parser.add_argument('-ch', '--checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.pth')


def _eval(path_to_checkpoint_file, path_to_data_dir, path_to_log_dir, lmdb_file, number_of_images_to_evaluate):
    path_to_eval_lmdb_dir = os.path.join(path_to_data_dir, lmdb_file)
    model = Model()
    model.restore(path_to_checkpoint_file)
    model.cpu()
    print(f'Evaluate {path_to_checkpoint_file} on {path_to_eval_lmdb_dir}')
    results = AltEvaluator(path_to_eval_lmdb_dir, number_of_images_to_evaluate).evaluate(model)

    return results

    # export_evaluate_to_data_dir(path_to_data_dir, results, get_model_version(path_to_checkpoint_file))


def get_model_version(path_to_checkpoint_file):
    start = path_to_checkpoint_file.find("-")+1
    end = path_to_checkpoint_file.find(".pth")
    substring = path_to_checkpoint_file[start:end]
    return substring 


def export_evaluate_to_data_dir(data_dir, data, model_version):
    evaluate_dir = f"{data_dir}/evaluate"
    Path(evaluate_dir).mkdir(parents=True, exist_ok=True)
    export_to_json(evaluate_dir, data, model_version)


def export_to_json(path_to_log_dir, data, model_version):
    with open(f'{path_to_log_dir}/data-{model_version}.json', 'w') as f:
        json.dump(data, f)


def main(args):
    number_of_images_to_evaluate = args.num_images

    path_to_data_dir = args.data_dir
    path_to_checkpoint_file = args.checkpoint
    log_dir = args.logdir
    lmdb_file = args.lmdb
    get_model_version(path_to_checkpoint_file)
    print('Start evaluate')
    results = _eval(path_to_checkpoint_file, path_to_data_dir, log_dir, lmdb_file, number_of_images_to_evaluate)
    print('Done')
    return results


if __name__ == '__main__':
    main(parser.parse_args())
