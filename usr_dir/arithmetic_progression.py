from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import requests

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
import random
import string

total_length = 40
observed_range = [i for i in range(0, 5000)]
shrinked_range = [i for i in range(0, 2000)]

@registry.register_problem
class ArithmeticProgressionInterpolation(text_problems.Text2TextProblem):
    """Mathematical language understanding, see arxiv.org/abs/1812.02825."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def test_difference(self):
        return [17, 21, 32, 45, 57, 61, 76, 80, 89, 92, 102, 9, 112, 124, 136, 149, 158, 167, 173, 182, 190]

    @property
    def train_difference(self):
        return [i for i in range(1, 200) if i not in self.test_difference]

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 2,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 2,
        }]

    def generate_seq(self, first_term, difference):
        seq = ""
        current_term = first_term
        seq += str(current_term)+" "
        while len(seq) < 41:
            current_term += difference
            seq += str(current_term) + " "
        seq = seq[:40]
        return seq[:20], seq[20:]

    def test_eqn(self, shard):
        if shard % 2 == 0:
            first_term = random.choice(shrinked_range)
            difference = random.choice(self.train_difference)
            res, res2 = self.generate_seq(first_term, difference)
        elif shard % 2 == 1:
            first_term = random.choice(shrinked_range)
            difference = random.choice(self.test_difference)
            res, res2 = self.generate_seq(first_term, difference)

        return res, res2

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            for first_term in observed_range:
                for difference in self.train_difference:
                    enc, dec = self.generate_seq(first_term, difference)
                    yield {"inputs": enc, "targets": dec}
        else:
            num_data = int(2*1e3)
            for num in range(num_data):
                enc, dec = self.test_eqn(num)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }



observed_difference = [i for i in range(1, 51)]
unobserved_difference1 = [i for i in range(51, 101)]
unobserved_difference2 = [i for i in range(101, 151)]
unobserved_difference3 = [i for i in range(151, 200)]


@registry.register_problem
class ArithmeticProgression(text_problems.Text2TextProblem):
    """Mathematical language understanding, see arxiv.org/abs/1812.02825."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 2,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 4,
        }]

    def generate_seq(self, first_term, difference):
        seq = ""
        current_term = first_term
        seq += str(current_term)+" "
        while len(seq)<41:
            current_term += difference
            seq += str(current_term) + " "
        seq = seq[:40]
        return seq[:20], seq[20:]

    def test_eqn(self, shard):
        if shard % 4 == 0:
            first_term = random.choice(shrinked_range)
            difference = random.choice(observed_difference)
            res, res2 = self.generate_seq(first_term, difference)
        elif shard % 4 == 1:
            first_term = random.choice(shrinked_range)
            difference = random.choice(unobserved_difference1)
            res, res2 = self.generate_seq(first_term, difference)
        elif shard % 4 == 2:
            first_term = random.choice(shrinked_range)
            difference = random.choice(unobserved_difference2)
            res, res2 = self.generate_seq(first_term, difference)
        elif shard % 4 == 3:
            first_term = random.choice(shrinked_range)
            difference = random.choice(unobserved_difference3)
            res, res2 = self.generate_seq(first_term, difference)


        return res, res2

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            for first_term in observed_range:
                for difference in observed_difference:
                    enc, dec = self.generate_seq(first_term, difference)
                    yield {"inputs": enc, "targets": dec}
        else:
            num_data = int(4*1e3)
            for num in range(num_data):
                enc, dec = self.test_eqn(num)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }


@registry.register_problem
class ArithmeticProgressionSanity(text_problems.Text2TextProblem):
    """Mathematical language understanding, see arxiv.org/abs/1812.02825."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 2,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    def generate_seq(self, first_term, difference):
        seq = ""
        current_term = first_term
        seq += str(current_term)+" "
        while len(seq)<41:
            current_term += difference
            seq += str(current_term) + " "
        seq = seq[:40]
        return seq[:20], seq[20:]

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            for first_term in observed_range:
                for difference in observed_difference + unobserved_difference:
                    enc, dec = self.generate_seq(first_term, difference)
                    yield {"inputs": enc, "targets": dec}
        else:
            for num in range(int(1e4)):
                first_term = random.choice(observed_range)
                difference = random.choice(observed_difference + unobserved_difference)
                enc, dec = self.generate_seq(first_term, difference)
                yield {"inputs": enc, "targets": dec}
