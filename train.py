import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable
from dataset import Dataset
from model import Model
from evaluator import Evaluator

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')
parser.add_argument('-l', '--logdir', default='./logs', help='directory to write logs')
parser.add_argument('-r', '--restore_checkpoint', default=None,
                    help='path to restore checkpoint, e.g. ./logs/model-100.tar')
parser.add_argument('-b', '--batch_size', default=32, type=int,  help='Default 32')
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Default 1e-2')
parser.add_argument('-p', '--patience', default=100, type=int, help='Default 100, set -1 to train infinitely')
parser.add_argument('-ds', '--decay_steps', default=10000, type=int, help='Default 10000')
parser.add_argument('-dr', '--decay_rate', default=0.9, type=float, help='Default 0.9')


def _loss(length_logits, digits_logits, length_labels, digits_labels):
    length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, length_labels)
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[0], digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[1], digits_labels[1])
    digit3_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[2], digits_labels[2])
    digit4_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[3], digits_labels[3])
    digit5_cross_entropy = torch.nn.functional.cross_entropy(digits_logits[4], digits_labels[4])
    loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy + digit3_cross_entropy + digit4_cross_entropy + digit5_cross_entropy
    return loss


def _adjust_learning_rate(optimizer, step, initial_lr, decay_steps, decay_rate):
    lr = initial_lr * (decay_rate ** (step // decay_steps))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def _train(path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_log_dir,
           path_to_restore_checkpoint_file, training_options):
    batch_size = training_options['batch_size']
    initial_learning_rate = training_options['learning_rate']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    step = 0
    patience = initial_patience
    best_accuracy = 0.0
    duration = 0.0

    model = Model()
    model.cuda()
    if path_to_restore_checkpoint_file is not None:
        assert os.path.isfile(path_to_restore_checkpoint_file), '%s not found' % path_to_restore_checkpoint_file
        step = model.load(path_to_restore_checkpoint_file)
        print 'Model restored from file: %s' % path_to_restore_checkpoint_file

    train_loader = torch.utils.data.DataLoader(Dataset(path_to_train_lmdb_dir),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=2, pin_memory=True)
    evaluator = Evaluator(path_to_val_lmdb_dir)
    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate)

    path_to_losses_npy_file = os.path.join(path_to_log_dir, 'losses.npy')
    if os.path.isfile(path_to_losses_npy_file):
        losses = np.load(path_to_losses_npy_file)
    else:
        losses = np.empty([0], dtype=np.float32)

    while True:
        for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
            start_time = time.time()
            images, length_labels, digits_labels = (Variable(images.cuda()),
                                                    Variable(length_labels.cuda()),
                                                    [Variable(digit_labels.cuda()) for digit_labels in digits_labels])
            length_logits, digits_logits = model.train()(images)
            loss = _loss(length_logits, digits_logits, length_labels, digits_labels)

            learning_rate = _adjust_learning_rate(optimizer, step=step, initial_lr=initial_learning_rate,
                                                  decay_steps=training_options['decay_steps'],
                                                  decay_rate=training_options['decay_rate'])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1
            duration += time.time() - start_time

            if step % num_steps_to_show_loss == 0:
                examples_per_sec = batch_size * num_steps_to_show_loss / duration
                duration = 0.0
                print '=> %s: step %d, loss = %f, learning_rate = %f (%.1f examples/sec)' % (
                    datetime.now(), step, loss.data[0], learning_rate, examples_per_sec)

            if step % num_steps_to_check != 0:
                continue

            losses = np.append(losses, loss.cpu().data.numpy())
            np.save(path_to_losses_npy_file, losses)

            print '=> Evaluating on validation dataset...'
            accuracy = evaluator.evaluate(model)
            print '==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy)

            if accuracy > best_accuracy:
                path_to_checkpoint_file = model.save(path_to_log_dir, step=step, maximum=2)
                print '=> Model saved to file: %s' % path_to_checkpoint_file
                patience = initial_patience
                best_accuracy = accuracy
            else:
                patience -= 1

            print '=> patience = %d' % patience
            if patience == 0:
                return


def main(args):
    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    path_to_log_dir = args.logdir
    path_to_restore_checkpoint_file = args.restore_checkpoint
    training_options = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate
    }

    if not os.path.exists(path_to_log_dir):
        os.makedirs(path_to_log_dir)

    print 'Start training'
    _train(path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_log_dir,
           path_to_restore_checkpoint_file, training_options)
    print 'Done'


if __name__ == '__main__':
    main(parser.parse_args())
