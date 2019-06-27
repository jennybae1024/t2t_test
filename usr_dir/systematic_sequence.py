import re
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import numpy as np
import random
import string
import tensorflow as tf


class AlgorithmicSequence(text_problems.Text2TextProblem):
    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return True

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def num_train(self):
        return 100000

    @property
    def num_eval(self):
        return 2000

    @property
    def num_test(self):
        return 2000

    def rule(self):
        raise NotImplementedError()

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        raise NotImplementedError()


@registry.register_problem
class AlgorithmicFibonazziSequence(AlgorithmicSequence):

    def generate_initial_values(self, shard):
        if shard == 0:
            a = np.random.randint(0, 4000)
            b = np.random.randint(0, 4000)
        elif shard == 1:
            a = np.random.randint(4000, 6000)
            b = np.random.randint(4000, 6000)
        return a, b


    def generate_seq(self, first_term, second_term):
        seq = ""
        temp1 = first_term
        temp2 = second_term
        seq += str(temp1)+" "+str(temp2)+" "
        while len(seq) < 41:
            next_temp2 = temp2 + temp1
            temp1 = temp2
            temp2 = next_temp2
            seq += str(temp2) + " "
        seq = seq[:40]
        return seq[:20], seq[20:]


    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            num_data = self.num_train
            for num in range(num_data):
                a, b = self.generate_initial_values(0)
                enc, dec = self.generate_seq(a, b)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }

        elif dataset_split == problem.DatasetSplit.EVAL:
            num_data = self.num_eval
            for num in range(num_data):
                a, b = self.generate_initial_values(1)
                enc, dec = self.generate_seq(a, b)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }

        elif dataset_split == problem.DatasetSplit.TEST:
            num_data = self.num_test
            for num in range(num_data):
                for shard in range(2):
                    a, b = self.generate_initial_values(shard)
                    enc, dec = self.generate_seq(a, b)
                    yield {
                        "inputs": enc,
                        "targets": dec,
                    }

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 2,
        }]


    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)



