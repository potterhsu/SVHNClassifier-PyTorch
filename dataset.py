import lmdb
import numpy as np
import torch.utils.data as data
from PIL import Image

from . import example_pb2


class Dataset(data.Dataset):
    def __init__(self, path_to_lmdb_dir, transform):
        self._path_to_lmdb_dir = path_to_lmdb_dir
        self._reader = lmdb.open(path_to_lmdb_dir, lock=False)
        with self._reader.begin() as txn:
            self._length = txn.stat()['entries']
            self._keys = self._keys = [key for key, _ in txn.cursor()]
        self._transform = transform

    def __len__(self):
        return self._length

    def __getitem__(self, index):
        if isinstance(index, slice):
            result = []
            for i in range(index.start, index.stop):
                path = self._keys[i]

                with self._reader.begin() as txn:
                    value = txn.get(path)

                example = example_pb2.Example()
                example.ParseFromString(value)

                image = np.frombuffer(example.image, dtype=np.uint8)
                image = image.reshape([64, 64, 3])
                image = Image.fromarray(image)
                image = self._transform(image)

                length = example.length
                digits = example.digits

                result.append((image, length, digits, path))
            return result
        else:
            path = self._keys[index]

            with self._reader.begin() as txn:
                value = txn.get(path)

            example = example_pb2.Example()
            example.ParseFromString(value)

            image = np.frombuffer(example.image, dtype=np.uint8)
            image = image.reshape([64, 64, 3])
            image = Image.fromarray(image)
            image = self._transform(image)

            length = example.length
            digits = example.digits

            return image, length, digits, path
