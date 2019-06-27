from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import io
from contextlib import redirect_stdout

import string
import io
from contextlib import redirect_stdout
import numpy as np

from random import choice, random, randint


import os
import tarfile
import requests

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry

import tensorflow as tf
import string





GO_SYMBOL = 'G'
PAD_SYMBOL = '_'
LETTERS = string.digits + string.ascii_lowercase + ' +-=():<>\n'
SYMBOLS = [GO_SYMBOL, PAD_SYMBOL] + list(LETTERS)
#SYMBOLS = [PAD_SYMBOL, GO_SYMBOL] + list(LETTERS)
SYMBOL_TO_IDX = dict((l, i) for i, l in enumerate(SYMBOLS))

INPUT_SEQ_LEN = 70
OUTPUT_SEQ_LEN = 6

MAX_NUM_LENGTH = 2
MAX_PROGRAM_LENGTH = 3

COMPARATORS = ('<', '>')
OPERATORS = ('+', '-')
VARIABLE_NAMES = list('abcdefgh')


def if_operation(variables, nesting, difficulty):
    compare_variable = choice(list(variables))
    comparator = choice(COMPARATORS)
    compare_value = random_digit(difficulty)
    code = 'if {0}{1}{2}:'.format(compare_variable,
                                  comparator,
                                  compare_value)
    nesting += 1
    return code, nesting


def assign_operation(variables, nesting, num_len):
    variable = choice(VARIABLE_NAMES)
    variables.add(variable)
    value = random_digit(num_len)
    code = '{0}={1}'.format(variable, value)
    return code, nesting


def add_or_sub_operation(variables, nesting, num_len):
    variable = choice(list(variables))
    operator = choice(OPERATORS)
    value = random_digit(num_len)
    if random() < 0.5:
        code = '{0}{1}={2}'.format(variable, operator, value)
    else:
        variable2 = choice(list(variables))
        code = '{0}={1}{2}{3}'.format(variable, variable2, operator, value)

    return code, nesting


def print_operation(variables, nesting, num_len):
    operator = choice(OPERATORS)
    code = 'print({0})'.format(operator.join(list(variables)))
    return code, nesting


OPERATIONS = (add_or_sub_operation, if_operation, assign_operation)


def generate_program(num_len, length):
    variables = set()
    nesting = 0

    lines = []
    lines.append(assign_operation(variables, nesting, num_len)[0])

    if length > 0:
        num_lines = randint(1, length)
        for i in range(num_lines):
            if num_lines <= 1:
                operation = add_or_sub_operation
            elif nesting == 0:
                operation = choice(OPERATIONS)
            else:
                operation = choice((add_or_sub_operation, if_operation))

            code, new_nesting = operation(variables, nesting, num_len)
            lines.append(''.join(['  '] * nesting) + code)
            if nesting == new_nesting and random() < 0.5:
                nesting -= 1
            nesting = new_nesting

        if nesting > 0:
            code, new_nesting = add_or_sub_operation(variables, nesting, num_len)
            lines.append(''.join(['  '] * nesting) + code)

    lines.append(print_operation(variables, nesting, num_len)[0])

    return '\n'.join(lines)


def random_digit(difficulty):
    size = 10 ** randint(1, difficulty)
    if difficulty > 1:
        return randint(-size, size)
    else:
        return randint(0, size)





def encode_sequences(letter_sequences, symbol_to_idx, sequence_len, pad_symbol=None, go_symbol=None,
                     pad_beginning=True, reverse=False, ):
    """
    Given a set of symbols and their index/label encoded the given
    list of string sequences as numeric sequences.
    """

    pad_idx = symbol_to_idx[pad_symbol]

    if go_symbol is None:
        go_idx = None
    else:
        go_idx = symbol_to_idx[go_symbol]

    assert sequence_len >= len(max(letter_sequences, key=len)) + 0 if go_idx is None else 1

    encoded_sequences = np.full((len(letter_sequences), sequence_len),
                                fill_value=pad_idx,
                                dtype=np.int32)

    for i, sequence in enumerate(letter_sequences):

        idxs = [symbol_to_idx[symbol] for symbol in sequence]

        if reverse:
            idxs = idxs[::-1]

        # Insert the idx of the GO symbol to the end of the sequence.
        if go_idx is not None:
            idxs.append(go_idx)

        if pad_beginning:
            encoded_sequences[i, -len(idxs):] = idxs
        else:
            encoded_sequences[i, :len(idxs)] = idxs

    return encoded_sequences


def decode_output_sequences(sequences, symbols):
    """
    Args:
        sequences: ndarray
            Shape: (num_seq, time_steps, output_size)
        symbols: [str]

    Returns:
        decoded_sequences: [str]
    """

    decoded_sequences = []
    for sequence in np.argmax(sequences, axis=2):
        decoded_sequences.append(''.join(symbols[idx] for idx in sequence))
    return decoded_sequences


def dense_to_one_hot(labels_dense, num_classes):
    """
    Convert class labels from scalars to one-hot vectors.

    Args:
        labels_dense: array, 1D or 2D, int32
            Shape: (num_samples) or (num_sequences, sequence_len)
        num_classes: int

    Returns:
        labels_one_hot: array, 2D or 3D, float32
            Shape: (num_samples, num_classes) or
            (num_sequences, sequence_len, num_classes)
    """

    assert labels_dense.ndim == 1 or labels_dense.ndim == 2
    assert labels_dense.dtype == np.int32

    if labels_dense.ndim == 1:
        num_sequences = 0
        sequence_len = labels_dense.shape
    else:
        num_sequences, sequence_len = labels_dense.shape

    labels_dense = labels_dense.reshape(-1)
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes), dtype=np.float32)
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1

    if num_sequences > 0:
        labels_one_hot = labels_one_hot.reshape((num_sequences, sequence_len, num_classes))

    return labels_one_hot



@registry.register_problem
class LearningToExecute(text_problems.Text2TextProblem):
    """Mathematical language understanding, see arxiv.org/abs/1812.02825."""

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    @property
    def dataset_splits(self):
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 4,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }, {
            "split": problem.DatasetSplit.TEST,
            "shards": 1,
        }]

    @property
    def is_generate_per_split(self):
        return True

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        num_data = int(1e6)

        if dataset_split == problem.DatasetSplit.TRAIN:
            for num in range(num_data):
                results = []
                program = generate_program(2, 5)
                with io.StringIO() as buf, redirect_stdout(buf):
                    exec(program)
                    results.append(buf.getvalue()[:-1])

                yield {"inputs": program, "targets": results[0]}
        else:
            for num in range(1000):
                results = []
                program = generate_program(2, 5)
                with io.StringIO() as buf, redirect_stdout(buf):
                    exec(program)
                    results.append(buf.getvalue()[:-1])


@registry.register_problem
class LearningToExecute36(LearningToExecute):
    """Mathematical language understanding, see arxiv.org/abs/1812.02825."""

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        num_data = int(1e6)

        if dataset_split == problem.DatasetSplit.TRAIN:
            for num in range(num_data):
                results = []
                program = generate_program(3, 6)
                with io.StringIO() as buf, redirect_stdout(buf):
                    exec(program)
                    results.append(buf.getvalue()[:-1])

                yield {"inputs": program, "targets": results[0]}
        else:
            for num in range(10000):
                results = []
                program = generate_program(3, 6)
                with io.StringIO() as buf, redirect_stdout(buf):
                    exec(program)
                    results.append(buf.getvalue()[:-1])

                yield {"inputs": program, "targets": results[0]}

                yield {"inputs": program, "targets": results[0]}






