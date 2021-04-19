import argparse
import os
import numpy as np
from visdom import Visdom

parser = argparse.ArgumentParser()
parser.add_argument('-l', '--logdir', default='./logs', help='directory to read logs')


def _visualize(path_to_log_dir):
    losses = np.load(os.path.join(path_to_log_dir, 'losses.npy'))
    test_losses = np.load(os.path.join(path_to_log_dir, 'test_losses.npy'))

    viz = Visdom()
    viz.line(losses)
    viz.line(test_losses)


def main(args):
    path_to_log_dir = args.logdir
    _visualize(path_to_log_dir)


if __name__ == '__main__':
    main(parser.parse_args())
