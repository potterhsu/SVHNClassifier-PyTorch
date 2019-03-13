import argparse
import glob
import os
import random

import h5py
import lmdb
import numpy as np
from PIL import Image

import example_pb2
from meta import Meta

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to SVHN (format 1) folders and write the converted files')


class ExampleReader(object):
    def __init__(self, path_to_image_files):
        self._path_to_image_files = path_to_image_files
        self._num_examples = len(self._path_to_image_files)
        self._example_pointer = 0

    @staticmethod
    def _get_attrs(digit_struct_mat_file, index):
        """
        Returns a dictionary which contains keys: label, left, top, width and height, each key has multiple values.
        """
        attrs = {}
        f = digit_struct_mat_file
        item = f['digitStruct']['bbox'][index].item()
        for key in ['label', 'left', 'top', 'width', 'height']:
            attr = f[item][key]
            values = [f[attr[i].item()][0][0]
                      for i in range(len(attr))] if len(attr) > 1 else [attr[0][0]]
            attrs[key] = values
        return attrs

    @staticmethod
    def _preprocess(image, bbox_left, bbox_top, bbox_width, bbox_height):
        cropped_left, cropped_top, cropped_width, cropped_height = (int(round(bbox_left - 0.15 * bbox_width)),
                                                                    int(round(bbox_top - 0.15 * bbox_height)),
                                                                    int(round(bbox_width * 1.3)),
                                                                    int(round(bbox_height * 1.3)))
        image = image.crop([cropped_left, cropped_top, cropped_left + cropped_width, cropped_top + cropped_height])
        image = image.resize([64, 64])
        return image

    def read_and_convert(self, digit_struct_mat_file):
        """
        Read and convert to example, returns None if no data is available.
        """
        if self._example_pointer == self._num_examples:
            return None
        path_to_image_file = self._path_to_image_files[self._example_pointer]
        index = int(path_to_image_file.split('/')[-1].split('.')[0]) - 1
        self._example_pointer += 1

        attrs = ExampleReader._get_attrs(digit_struct_mat_file, index)
        label_of_digits = attrs['label']
        length = len(label_of_digits)
        if length > 5:
            # skip this example
            return self.read_and_convert(digit_struct_mat_file)

        digits = [10, 10, 10, 10, 10]   # digit 10 represents no digit
        for idx, label_of_digit in enumerate(label_of_digits):
            digits[idx] = int(label_of_digit if label_of_digit != 10 else 0)    # label 10 is essentially digit zero

        attrs_left, attrs_top, attrs_width, attrs_height = map(lambda x: [int(i) for i in x], [attrs['left'], attrs['top'], attrs['width'], attrs['height']])
        min_left, min_top, max_right, max_bottom = (min(attrs_left),
                                                    min(attrs_top),
                                                    max(map(lambda x, y: x + y, attrs_left, attrs_width)),
                                                    max(map(lambda x, y: x + y, attrs_top, attrs_height)))
        center_x, center_y, max_side = ((min_left + max_right) / 2.0,
                                        (min_top + max_bottom) / 2.0,
                                        max(max_right - min_left, max_bottom - min_top))
        bbox_left, bbox_top, bbox_width, bbox_height = (center_x - max_side / 2.0,
                                                        center_y - max_side / 2.0,
                                                        max_side,
                                                        max_side)
        image = np.array(ExampleReader._preprocess(Image.open(path_to_image_file), bbox_left, bbox_top, bbox_width, bbox_height)).tobytes()

        example = example_pb2.Example()
        example.image = image
        example.length = length
        example.digits.extend(digits)
        return example


def convert_to_lmdb(path_to_dataset_dir_and_digit_struct_mat_file_tuples,
                    path_to_lmdb_dirs, choose_writer_callback):
    num_examples = []
    writers = []

    for path_to_lmdb_dir in path_to_lmdb_dirs:
        num_examples.append(0)
        writers.append(lmdb.open(path_to_lmdb_dir, map_size=10*1024*1024*1024))

    for path_to_dataset_dir, path_to_digit_struct_mat_file in path_to_dataset_dir_and_digit_struct_mat_file_tuples:
        path_to_image_files = glob.glob(os.path.join(path_to_dataset_dir, '*.png'))
        total_files = len(path_to_image_files)
        print('%d files found in %s' % (total_files, path_to_dataset_dir))

        with h5py.File(path_to_digit_struct_mat_file, 'r') as digit_struct_mat_file:
            example_reader = ExampleReader(path_to_image_files)
            block_size = 10000

            for i in range(0, total_files, block_size):
                txns = [writer.begin(write=True) for writer in writers]

                for offset in range(block_size):
                    idx = choose_writer_callback(path_to_lmdb_dirs)
                    txn = txns[idx]

                    example = example_reader.read_and_convert(digit_struct_mat_file)
                    if example is None:
                        break

                    str_id = '{:08}'.format(num_examples[idx] + 1)
                    txn.put(str_id.encode(), example.SerializeToString())
                    num_examples[idx] += 1

                    index = i + offset
                    path_to_image_file = path_to_image_files[index]
                    print('(%d/%d) %s' % (index + 1, total_files, path_to_image_file))

                [txn.commit() for txn in txns]

    for writer in writers:
        writer.close()

    return num_examples


def create_lmdb_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_lmdb_meta_file):
    print('Saving meta file to %s...' % path_to_lmdb_meta_file)
    meta = Meta()
    meta.num_train_examples = num_train_examples
    meta.num_val_examples = num_val_examples
    meta.num_test_examples = num_test_examples
    meta.save(path_to_lmdb_meta_file)


def main(args):
    path_to_train_dir = os.path.join(args.data_dir, 'train')
    path_to_test_dir = os.path.join(args.data_dir, 'test')
    path_to_extra_dir = os.path.join(args.data_dir, 'extra')
    path_to_train_digit_struct_mat_file = os.path.join(path_to_train_dir, 'digitStruct.mat')
    path_to_test_digit_struct_mat_file = os.path.join(path_to_test_dir, 'digitStruct.mat')
    path_to_extra_digit_struct_mat_file = os.path.join(path_to_extra_dir, 'digitStruct.mat')

    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    path_to_test_lmdb_dir = os.path.join(args.data_dir, 'test.lmdb')
    path_to_lmdb_meta_file = os.path.join(args.data_dir, 'lmdb_meta.json')

    for path_to_dir in [path_to_train_lmdb_dir, path_to_val_lmdb_dir, path_to_test_lmdb_dir]:
        assert not os.path.exists(path_to_dir), 'LMDB directory %s already exists' % path_to_dir

    print('Processing training and validation data...')
    [num_train_examples, num_val_examples] = convert_to_lmdb([(path_to_train_dir, path_to_train_digit_struct_mat_file),
                                                              (path_to_extra_dir, path_to_extra_digit_struct_mat_file)],
                                                             [path_to_train_lmdb_dir, path_to_val_lmdb_dir],
                                                             lambda paths: 0 if random.random() > 0.1 else 1)
    print('Processing test data...')
    [num_test_examples] = convert_to_lmdb([(path_to_test_dir, path_to_test_digit_struct_mat_file)],
                                          [path_to_test_lmdb_dir],
                                          lambda paths: 0)

    create_lmdb_meta_file(num_train_examples, num_val_examples, num_test_examples, path_to_lmdb_meta_file)

    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
