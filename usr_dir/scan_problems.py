# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tarfile
import requests

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import registry
from tensor2tensor.utils import mlperf_log
from . import scan_grammar
import tensorflow as tf
import random
import numpy as np
import string
import nltk
import six
from nltk.parse.generate import generate
import copy
import pdb


def scan_translate(commands):

    if 'and' in commands:
        x1 = commands.split("and")[0].strip()
        x2 = commands.split("and")[1].strip()
        return scan_translate(x1) + scan_translate(x2)
    if 'after' in commands:
        x1 = commands.split("after")[0].strip()
        x2 = commands.split("after")[1].strip()
        return scan_translate(x2) + scan_translate(x1)
    if 'twice' in commands:
        x = commands.split("twice")[0].strip()
        return scan_translate(x) + scan_translate(x)
    if 'thrice' in commands:
        x = commands.split("thrice")[0].strip()
        return scan_translate(x) + scan_translate(x) + scan_translate(x)
    if 'around left' in commands:
        x = commands.split("around left")[0].strip()
        if x == 'turn':
            return "".join(['LTURN ' for _ in range(4)])
        else:
            return "".join(['LTURN ' + scan_translate(x) for _ in range(4)])
    if 'around right' in commands:
        x = commands.split("around right")[0].strip()
        if x == 'turn':
            return "".join(['RTURN ' for _ in range(4)])
        else:
            return "".join(['RTURN ' + scan_translate(x) for _ in range(4)])
    if 'opposite left' in commands:
        x = commands.split("opposite left")[0].strip()
        if x == 'turn':
            return "".join(['LTURN ' for _ in range(2)])
        else:
            return "LTURN LTURN " + scan_translate(x)
    if 'opposite right' in commands:
        x = commands.split("opposite right")[0].strip()
        if x == 'turn':
            return "".join(['RTURN ' for _ in range(2)])
        else:
            return "RTURN RTURN " + scan_translate(x)
    if 'left' in commands and 'turn' not in commands:
        return "LTURN " + scan_translate(commands.split("left")[0].strip())
    if 'right' in commands and 'turn' not in commands:
        return "RTURN " + scan_translate(commands.split("right")[0].strip())
    if 'walk'==commands:
        return "WALK "
    if 'look'==commands:
        return "LOOK "
    if 'run'==commands:
        return "RUN "
    if 'jump'==commands:
        return "JUMP "
    if 'turn left'==commands:
        return "LTURN "
    if 'turn right'==commands:
        return "RTURN "


class GrammarTextEncoder(text_encoder.TextEncoder):
    """Encoder based on a user-supplied vocabulary (file or list)."""

    def __init__(self,
                 grammar,
                 reverse=False,
                 replace_oov=None,
                 num_reserved_ids=2):

        super(GrammarTextEncoder, self).__init__(num_reserved_ids=num_reserved_ids)
        self._reverse = reverse
        self._replace_oov = replace_oov
        self._grammar = grammar
        self._prod_to_id = self._grammar.prod_map
        self._id_to_prod = ["<pad>", "<EOS>"] + self._grammar.GCFG.productions()

    def encode(self, s):
        """Converts a space-separated string of tokens to a list of ids."""
        sentence = s
        tokens = sentence.strip().split()
        trees = [i for i in self._grammar.parser.parse(tokens)][0]
        ret = [self._prod_to_id[prod] for prod in trees.productions()]
        return ret[::-1] if self._reverse else ret

    def decode(self, ids, strip_extraneous=False):
        def prods_to_eq(prods):
            seq = [prods[0].lhs()]
            for prod in prods:
                if str(prod) == '<pad>':
                    break
                for ix, s in enumerate(seq):
                    if s == prod.lhs():
                        seq = seq[:ix] + list(prod.rhs()) + seq[ix + 1:]
                        break
            try:
                return ' '.join(seq)
            except:
                return ''
        prods = [self._id_to_prod[id] for id in ids]
        return prods_to_eq(prods)

    def decode_list(self, ids):
        seq = reversed(ids) if self._reverse else ids
        return [self._safe_id_to_token(i) for i in seq]

    @property
    def vocab_size(self):
        return len(self._prod_to_id)

    def _safe_id_to_token(self, idx):
        return self._prod_to_id.get(idx, "ID_%d" % idx)

    def store_to_file(self, filename):
        with tf.gfile.Open(filename, "w") as f:
            for i in range(len(self._prod_to_token)):
                f.write(self._prod_to_token[i] + "\n")


def text2text_generate_encoded_without_eos(sample_generator,
                                           vocab,
                                           targets_vocab=None,
                                           has_inputs=True,
                                           inputs_prefix="",
                                           targets_prefix=""):
    """Encode Text2Text samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
        if has_inputs:
            print(sample["inputs"])
            sample["inputs"] = vocab.encode(inputs_prefix + sample["inputs"])
        sample["targets"] = targets_vocab.encode(targets_prefix + sample["targets"])
        yield sample

def text2text_generate_encoded_grammar(sample_generator,
                                       vocab,
                                       targets_vocab=None,
                                       has_inputs=True,
                                       targets_prefix=""):
    """Encode Text2Text samples from the generator with the vocab."""
    targets_vocab = targets_vocab or vocab
    for sample in sample_generator:
        if has_inputs:
            sample["inputs"] = vocab.encode(sample["inputs"])
            sample["inputs"] += [0 for _ in range(9 - len(sample["inputs"]))]
            sample["inputs"] += [1]
        sample["targets"] = targets_vocab.encode(targets_prefix + sample["targets"])
        yield sample

@registry.register_problem
class AlgorithmicScanV1(text_problems.Text2TextProblem):

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
        elif dataset_split == problem.DatasetSplit.EVAL:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return text2text_generate_encoded_without_eos(generator, encoder,
                                                      has_inputs=self.has_inputs)

    @property
    def vocab_filename(self):
        return "vocab.algorithmic_scan_v1.32.tokens"

    @property
    def vocab_type(self):
        return text_problems.VocabType.TOKEN

    @property
    def approx_vocab_size(self):
        return 2**5

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 8,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 2,
        }]

    @property
    def is_generate_per_split(self):
        return False

    @property
    def batch_size_means_tokens(self):
        return False

    def example_reading_spec(self):
        data_fields = {"targets": tf.FixedLenFeature([49], tf.int64)}
        if self.has_inputs:
            data_fields["inputs"] = tf.FixedLenFeature([10], tf.int64)

        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def uniform_len(self, c, max_len):
        if len(c.split()) == max_len:
            return c
        return c + " " + " ".join(["<pad>" for _ in range(max_len-len(c.split()))])

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        random_c = copy.deepcopy(scan_grammar.CFL)
        random.shuffle(random_c)
        for c in random_c:
            # yield {"inputs": c, "targets": " ".join(scan_translate(c).split())}
            input = self.uniform_len(c + " <EOS>", 10)
            target = " ".join(scan_translate(c).split())
            target = self.uniform_len(target + " <EOS>", 49)
            yield {"inputs": input, "targets": target}

@registry.register_problem
class AlgorithmicScanGrammarV1(AlgorithmicScanV1):

    def feature_encoders(self, data_dir):
        encoder = self.get_or_create_vocab(data_dir, None, force_get=True)
        encoders = {"targets": encoder}
        if self.has_inputs:
            encoders["inputs"] = GrammarTextEncoder(scan_grammar)
        return encoders

    def generate_encoded_samples(self, data_dir, tmp_dir, dataset_split):
        if dataset_split == problem.DatasetSplit.TRAIN:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_TRAINING)
        elif dataset_split == problem.DatasetSplit.EVAL:
            mlperf_log.transformer_print(key=mlperf_log.PREPROC_TOKENIZE_EVAL)

        generator = self.generate_samples(data_dir, tmp_dir, dataset_split)
        encoder = GrammarTextEncoder(scan_grammar)
        target_encoder = self.get_or_create_vocab(data_dir, tmp_dir)
        return text2text_generate_encoded_grammar(generator, encoder, target_encoder,
                                                  has_inputs=self.has_inputs)

    def example_reading_spec(self):
        data_fields = {"targets": tf.FixedLenFeature([49], tf.int64)}
        if self.has_inputs:
            data_fields["inputs"] = tf.FixedLenFeature([10], tf.int64)

        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        random_c = copy.deepcopy(scan_grammar.CFL)
        random.shuffle(random_c)
        for c in random_c:
            # yield {"inputs": c, "targets": " ".join(scan_translate(c).split())}
            input = c
            target = " ".join(scan_translate(c).split())
            target = self.uniform_len(target + " <EOS>", 49)
            yield {"inputs": input, "targets": target}

@registry.register_problem
class AlgorithmicScanV2(AlgorithmicScanV1):
    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        random_c = copy.deepcopy(scan_grammar.CFL)
        random.shuffle(random_c)
        if dataset_split == problem.DatasetSplit.TRAIN:
            for c in random_c:
                target = " ".join(scan_translate(c).split())
                if len(target.split()) < 23:
                    input = self.uniform_len(c + " <EOS>", 10)
                    target = self.uniform_len(target + " <EOS>", 49)
                    yield {"inputs": input, "targets": target}

        else:
            for c in random_c:
                target = " ".join(scan_translate(c).split())
                if len(target.split()) > 22:
                    input = self.uniform_len(c + " <EOS>", 10)
                    target = self.uniform_len(target + " <EOS>", 49)
                    yield {"inputs": input, "targets": target}

@registry.register_problem
class AlgorithmicScanGrammarV2(AlgorithmicScanGrammarV1):

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        random_c = copy.deepcopy(scan_grammar.CFL)
        random.shuffle(random_c)

        if dataset_split == problem.DatasetSplit.TRAIN:
            for c in random_c:
                target = " ".join(scan_translate(c).split())
                if len(target.split()) < 23:
                    input = c
                    target = self.uniform_len(target + " <EOS>", 49)
                    yield {"inputs": input, "targets": target}

        else:
            for c in random_c:
                target = " ".join(scan_translate(c).split())
                if len(target.split()) > 22:
                    input = c
                    target = self.uniform_len(target + " <EOS>", 49)
                    yield {"inputs": input, "targets": target}




@registry.register_problem
class AlgorithmicScanReverseV1(AlgorithmicScanV1):

    def example_reading_spec(self):
        data_fields = {"targets": tf.FixedLenFeature([10], tf.int64)}
        if self.has_inputs:
            data_fields["inputs"] = tf.FixedLenFeature([49], tf.int64)

        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        random_c = copy.deepcopy(scan_grammar.CFL)
        random.shuffle(random_c)
        for c in random_c:
            # yield {"inputs": c, "targets": " ".join(scan_translate(c).split())}
            input = self.uniform_len(c + " <EOS>", 10)
            target = " ".join(scan_translate(c).split())
            target = self.uniform_len(target + " <EOS>", 49)
            yield {"inputs": target, "targets": input}

@registry.register_problem
class AlgorithmicScanReverseV2(AlgorithmicScanReverseV1):

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        random_c = copy.deepcopy(scan_grammar.CFL)
        random.shuffle(random_c)
        if dataset_split == problem.DatasetSplit.TRAIN:
            for c in random_c:
                target = " ".join(scan_translate(c).split())
                if len(target.split()) < 23:
                    input = self.uniform_len(c + " <EOS>", 10)
                    target = self.uniform_len(target + " <EOS>", 49)
                    yield {"inputs": target, "targets": input}

        else:
            for c in random_c:
                target = " ".join(scan_translate(c).split())
                if len(target.split()) > 22:
                    input = self.uniform_len(c + " <EOS>", 10)
                    target = self.uniform_len(target + " <EOS>", 49)
                    yield {"inputs": target, "targets": input}

# @registry.register_problem
# class AlgorithmicScanReverseGrammarV1(AlgorithmicScanV1):
#
#     def example_reading_spec(self):
#         data_fields = {"targets": tf.FixedLenFeature([10], tf.int64)}
#         if self.has_inputs:
#             data_fields["inputs"] = tf.FixedLenFeature([49], tf.int64)
#
#         data_items_to_decoders = None
#         return (data_fields, data_items_to_decoders)
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#         random_c = copy.deepcopy(ScanGrammar.CFL)
#         random.shuffle(random_c)
#         for c in random_c:
#             # yield {"inputs": c, "targets": " ".join(scan_translate(c).split())}
#             input = c
#             target = " ".join(scan_translate(c).split())
#             target = self.uniform_len(target + " <EOS>", 49)
#             mask = None
#             yield {"inputs": target, "targets": input, "mask": mask}
#
# @registry.register_problem
# class AlgorithmicScanReverseGrammarV2(AlgorithmicScanReverseGrammarV1):
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 1,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 1,
#         }]
#
#     @property
#     def is_generate_per_split(self):
#         return True
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#         random_c = copy.deepcopy(ScanGrammar.CFL)
#         random.shuffle(random_c)
#
#         if dataset_split == problem.DatasetSplit.TRAIN:
#             for c in random_c:
#                 target = " ".join(scan_translate(c).split())
#                 if len(target.split()) < 23:
#                     input = c
#                     target = self.uniform_len(target + " <EOS>", 49)
#                     yield {"inputs": target, "targets": input}
#
#         else:
#             for c in random_c:
#                 target = " ".join(scan_translate(c).split())
#                 if len(target.split()) > 22:
#                     input = c
#                     target = self.uniform_len(target + " <EOS>", 49)
#                     yield {"inputs": target, "targets": input}


