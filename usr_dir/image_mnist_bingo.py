# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import requests

from tensor2tensor.data_generators import problem, generator_utils, image_utils
from tensor2tensor.utils import registry

import tensorflow as tf
import random
import numpy as np
import string

import gzip
import copy
_MNIST_TRAIN_DATA_FILENAME = "train-images-idx3-ubyte.gz"
_MNIST_TRAIN_LABELS_FILENAME = "train-labels-idx1-ubyte.gz"
_MNIST_TEST_DATA_FILENAME = "t10k-images-idx3-ubyte.gz"
_MNIST_TEST_LABELS_FILENAME = "t10k-labels-idx1-ubyte.gz"

def _extract_mnist_images(filename, num_images):
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(28 * 28 * num_images)
        data = np.frombuffer(buf, dtype=np.uint8)
        data = data.reshape(num_images, 28, 28, 1)
    return data

def _extract_mnist_labels(filename, num_labels):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(num_labels)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def one2two(i):
    return int(i/3), i%3

def make_bingo_labels(num_list):
    possible_bingos = [[0,1,2], [3,4,5], [6,7,8], [0,3,6], [1,4,7], [2,5,8], [0,4,8], [2,4,6]]
    a = np.random.choice(num_list)
    res = np.random.choice(num_list, [3,3])
    inds = possible_bingos[np.random.choice([i for i in range(len(possible_bingos))])]
    for i in inds:
        row, col = one2two(i)
        res[row][col] = a
    return res

def make_non_bingo_labels(num_list):
    res = np.random.choice(num_list, [3,3])
    while(is_bingo(res)):
        res = np.random.choice(num_list, [3,3])
    return res

def is_bingo(bingo_label):
    row_bingo = is_row_bingo(bingo_label)
    col_bingo = is_col_bingo(bingo_label)
    diagonal_bingo = is_diagonal_bingo(bingo_label)
    return row_bingo or col_bingo or diagonal_bingo

def is_row_bingo(bingo_label):
    for i in bingo_label:
        exist = True
        temp = i[0]
        for j in i:
            if j != temp:
                exist = False
                break
        if exist:
            return exist
    return exist

def is_col_bingo(bingo_label):
    label2 = copy.deepcopy(bingo_label)
    return is_row_bingo(np.transpose(label2))

def is_diagonal_bingo(bingo_label):
    temp = bingo_label[0][0]
    for i in range(len(bingo_label)):
        exist = True
        if bingo_label[i][i] != temp:
            exist = False
            break
    if exist:
        return exist
    temp = bingo_label[len(bingo_label)-1][0]
    for i in range(len(bingo_label)):
        exist = True
        if bingo_label[len(bingo_label)-1-i][i] != temp:
            exist = False
            break
    if exist:
        return exist
    return exist

def make_bingo_image(image_by_label, square_labels):
    sq = []
    for row in square_labels:
        sq.append([])
        for label in row:
            sq[-1].append(random.choice(image_by_label[label]))
        sq[-1] = np.concatenate(sq[-1], axis = 0)
    sq = np.concatenate(sq, axis = 1)
    return sq, np.transpose(square_labels)

def mnist_grid_generator(tmp_dir, training, how_many, start_from=0):
    d = _MNIST_TRAIN_DATA_FILENAME if training else _MNIST_TEST_DATA_FILENAME
    l = _MNIST_TRAIN_LABELS_FILENAME if training else _MNIST_TEST_LABELS_FILENAME
    images = _extract_mnist_images(d, 60000 if training else 10000)
    labels = _extract_mnist_labels(l, 60000 if training else 10000)
    # Shuffle the data to make sure classes are well distributed.
    data = list(zip(images, labels))
    random.shuffle(data)
    images, labels = list(zip(*data))
    image_by_label = {}
    for i in range(len(images)):
        if image_by_label.get(labels[i]):
            image_by_label[labels[i]].append(images[i])
        else:
            image_by_label[labels[i]] = [images[i]]

    return image_by_label

@registry.register_problem
class ImageMnistBingo(image_utils.Image2ClassProblem):
    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return True

    @property
    def dataset_splits(self):
        """Splits of data to produce and number the output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": self.num_train_shards,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": self.num_eval_shards,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": self.num_test_shards,
        }]

    @property
    def num_channels(self):
        return 1

    @property
    def is_small(self):
        return True

    @property
    def num_classes(self):
        return 2

    @property
    def class_labels(self):
        return [str(c) for c in range(self.num_classes)]

    @property
    def already_shuffled(self):
        return False

    @property
    def num_train_shards(self):
        return 10

    @property
    def num_eval_shards(self):
        return 1

    @property
    def num_test_shards(self):
        return 1


    def preprocess_example(self, example, mode, unused_hparams):
        image = example["inputs"]
        image.set_shape([28*3, 28*3, 1])
        if not self._was_reversed:
            image = tf.image.per_image_standardization(image)
        example["inputs"] = image
        return example

    def generate_data(self, data_dir, tmp_dir, task_id=-1):
        filepath_fns = {
            problem.DatasetSplit.TRAIN: self.training_filepaths,
            problem.DatasetSplit.EVAL: self.dev_filepaths,
            problem.DatasetSplit.TEST: self.test_filepaths,
        }

        split_paths = [(split["split"], filepath_fns[split["split"]](
            data_dir, split["shards"], shuffled=self.already_shuffled))
                       for split in self.dataset_splits]
        all_paths = []
        for _, paths in split_paths:
            all_paths.extend(paths)

        if self.is_generate_per_split:
            for split, paths in split_paths:
                generator_utils.generate_files(
                    self.generate_samples(data_dir, tmp_dir, split), paths)
        else:
            generator_utils.generate_files(
                self.generate_samples(
                    data_dir, tmp_dir, problem.DatasetSplit.TRAIN), all_paths)

        generator_utils.shuffle_dataset(all_paths)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        images = []
        labels = []
        if dataset_split == problem.DatasetSplit.TRAIN:
            image_by_label = mnist_grid_generator(tmp_dir, True, 55000)
            for _ in range(100000):
                fn = random.choice([make_bingo_labels, make_non_bingo_labels])
                sq_l = fn([0, 1, 2, 3, 4])
                sq, sq_l = make_bingo_image(image_by_label, sq_l)
                images.append(sq)
                labels.append(1 if is_bingo(sq_l) else 0)

        elif dataset_split == problem.DatasetSplit.EVAL:
            image_by_label = mnist_grid_generator(tmp_dir, False, 10000)
            for _ in range(1000):
                fn = random.choice([make_bingo_labels, make_non_bingo_labels])
                sq_l = fn([0, 1, 2, 3, 4])
                sq, sq_l = make_bingo_image(image_by_label, sq_l)
                images.append(sq)
                labels.append(1 if is_bingo(sq_l) else 0)
        else:
            image_by_label = mnist_grid_generator(tmp_dir, False, 10000)
            for _ in range(1000):
                fn = random.choice([make_bingo_labels, make_non_bingo_labels])
                sq_l = fn([5,6,7,8,9])
                sq, sq_l = make_bingo_image(image_by_label, sq_l)
                images.append(sq)
                labels.append(1 if is_bingo(sq_l) else 0)

        return image_utils.image_generator(images, labels)

