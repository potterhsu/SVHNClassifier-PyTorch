import argparse
import os
from model import Model
from evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='path to evaluate checkpoint, e.g. ./logs/model-100.tar')
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')


def _eval(path_to_checkpoint_file, path_to_eval_lmdb_dir):
    model = Model()
    model.load(path_to_checkpoint_file)
    model.cuda()
    accuracy = Evaluator(path_to_eval_lmdb_dir).evaluate(model)
    print 'Evaluate %s on %s, accuracy = %f' % (path_to_checkpoint_file, path_to_eval_lmdb_dir, accuracy)


def main(args):
    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    path_to_test_lmdb_dir = os.path.join(args.data_dir, 'test.lmdb')
    path_to_checkpoint_file = args.checkpoint

    print 'Start evaluating'
    # _eval(path_to_checkpoint_file, path_to_train_lmdb_dir)
    # _eval(path_to_checkpoint_file, path_to_val_lmdb_dir)
    _eval(path_to_checkpoint_file, path_to_test_lmdb_dir)
    print 'Done'


if __name__ == '__main__':
    main(parser.parse_args())
