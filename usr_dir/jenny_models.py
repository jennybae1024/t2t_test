from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import re
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
import numpy as np



import copy
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers, modalities
from tensor2tensor.utils import registry, t2t_model
from tensor2tensor.models.transformer import transformer_tiny, transformer_base
from tensor2tensor.models.lstm import lstm
from tensor2tensor.models.research.universal_transformer import universal_transformer_tiny
from tensor2tensor.models.research.universal_transformer import update_hparams_for_universal_transformer
import tensorflow as tf


# def Inception_module(Input, C1=8, C3_R=2, C3=8, C5_R=2, C5=8, P3_R=2):
#     '''
#     C1, C3, C5: number of filters for the main convolutions
#     C3_R, C5_R, P3_R: number of filters for the dimensionality reduction convolutions
#     '''
#     out1 = tf.layers.conv2d(Input, C1, (1,1), activation=tf.nn.relu)
#     out2 = tf.layers.conv2d(tf.layers.conv2d(Input, C3_R, (1,1), activation=tf.nn.relu), C3, (3,3), padding='same', activation=tf.nn.relu)
#     out3 = tf.layers.conv2d(tf.layers.conv2d(Input, C5_R, (1,1), activation=tf.nn.relu), C5, (5,5), padding='same', activation=tf.nn.relu)
#     out4 = tf.layers.conv2d(tf.layers.max_pooling2d(Input, [3,3], 1,padding='same'), P3_R, (1,1), activation=tf.nn.relu)
#     Inception = tf.concat([out1, out2, out3, out4], axis=3)
#     return Inception

# @registry.register_model
# class GoogleNetInception(t2t_model.T2TModel):
#     def body(self, features):


def inception_module(input, C1, C3_R, C3, C5_R, C5, P3_R):
    out1 = tf.layers.conv2d(input, C1, (1,1), activation = tf.nn.relu, padding='same', data_format='channels_last')
    out2 = tf.layers.conv2d(tf.layers.conv2d(input, C3_R, (1,1), activation=tf.nn.relu,
                                             data_format='channels_last'),
                            C3, (3,3), padding='same', activation=tf.nn.relu, data_format='channels_last')
    out3 = tf.layers.conv2d(tf.layers.conv2d(input, C5_R, (1,1), activation=tf.nn.relu,
                                             data_format='channels_last'),
                            C5, (5,5), padding='same', activation=tf.nn.relu, data_format='channels_last')
    out4 = tf.layers.conv2d(tf.layers.max_pooling2d(input, (3,3),1, padding='same',
                                                    data_format='channels_last'),
                            P3_R, (1,1), activation=tf.nn.relu, data_format='channels_last')
    output = tf.concat([out1, out2, out3, out4], axis=3)
    return output

@registry.register_model
class GoogleNetInception(t2t_model.T2TModel):
    def body(self, features):
        inputs = features.get('inputs')
        out1 = tf.layers.conv2d(inputs, 64, [3,3], strides = 2, padding = 'same')
        out2 = tf.layers.max_pooling2d(out1, [3,3], strides = 2, padding = 'same')
        out3 = tf.layers.conv2d(out2, 192, [3,3], strides = 1, padding = 'same')
        out4 = tf.layers.max_pooling2d(out3,[3,3], strides = 1, padding = 'same')
        inception1 = inception_module(out4, C1=64, C3_R=96, C3=128, C5_R=16, C5=32, P3_R=32)
        inception2 = inception_module(inception1, C1=128, C3_R=128, C3=192, C5_R=32, C5=96, P3_R=64)
        out5 = tf.layers.max_pooling2d(inception2, (3,3), strides=2, data_format= 'channels_last')
        inception3 = inception_module(out5, C1=192, C3_R=96, C3=208, C5_R=16, C5=48, P3_R=64)
        output = inception_module(inception3, C1=160, C3_R=112, C3=224, C5_R=24, C5=64, P3_R=64)
        # output = tf.layers.dense(tf.layers.flatten(inception4), 10)

        return output

@registry.register_model
class BasicLogisticClassifier(t2t_model.T2TModel):
    """Seq to seq LSTM with attention."""

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        # if self._hparams.initializer == "orthogonal":
        #     raise ValueError("LSTM models fail with orthogonal initializer.")

        inputs = features.get("inputs")
        filters = self._hparams.hidden_size
        h1 = tf.layers.conv2d(inputs, filters, kernel_size=[3,3], strides=1, padding='valid')
        h2 = tf.layers.conv2d(tf.nn.relu(h1), filters, kernel_size=[3,3], strides=1, padding='valid')
        # dense = tf.layers.dense(h2, 10, activation=tf.nn.relu)
        return tf.layers.conv2d(tf.nn.relu(h2), filters, kernel_size=[3,3])



@registry.register_model
class BasicLM(t2t_model.T2TModel):
    """Seq to seq LSTM with attention."""

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        # if self._hparams.initializer == "orthogonal":
        #     raise ValueError("LSTM models fail with orthogonal initializer.")

        inputs = features.get("inputs")
        batch_size = common_layers.shape_list(inputs)[0]

        filters = self._hparams.hidden_size

        cells = []
        for i in range(self._hparams.num_hidden_layers):
            c = tf.nn.rnn_cell.LSTMCell(filters)
            cells.append(c)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell, common_layers.flatten4d3d(inputs),
                                           initial_state=initial_state)

        return tf.expand_dims(outputs, axis=2)


@registry.register_hparams
def logistic_tiny():
    """HParams for supervised attention problems."""
    hparams = common_hparams.basic_params1()
    hparams.batch_size = 100

    return hparams


@registry.register_problem
class TinyShakespeare(text_problems.Text2TextProblem):
    """Predict next character in Shakespeare texts."""

    @property
    def approx_vocab_size(self):
        return 2**13  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def vocab_type(self):
        """What kind of vocabulary to use.

        `VocabType`s:
          * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
            Must provide `self.approx_vocab_size`. Generates the vocabulary based on
            the training data. To limit the number of samples the vocab generation
            looks at, override `self.max_samples_for_vocab`. Recommended and
            default.
          * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
          * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
            vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
            will not be generated for you. The vocab file should be stored in
            `data_dir/` with the name specified by `self.vocab_filename`.

        Returns:
          VocabType constant
        """
        return text_problems.VocabType.CHARACTER

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split
        import os
        input_file = '/media/disk1/jennybae/data/t2t_data/tinyshakespeare/input.txt'

        with open(input_file, 'r') as f:
            data = f.read().strip().split('\n')
            data = [entity for entity in data if len(entity)>0]

        for line in data:
            yield {
                'inputs': line,
                'targets': line[1:]
                }



@registry.register_problem
class TinyShakespeareChar(text_problems.Text2TextProblem):
    """Predict next character of Shakespeare's script.
    Copy from class PoetryLines(text_problems.Text2TextProblem)"""

    @property
    def approx_vocab_size(self):
        return 2**13  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return False

    @property
    def dataset_splits(self):
        """Splits of data to produce and number of output shards for each."""
        # 10% evaluation data
        return [{
            "split": problem.DatasetSplit.TRAIN,
            "shards": 9,
        }, {
            "split": problem.DatasetSplit.EVAL,
            "shards": 1,
        }]

    @property
    def vocab_type(self):
        """What kind of vocabulary to use.

        `VocabType`s:
          * `SUBWORD`: `SubwordTextEncoder`, an invertible wordpiece vocabulary.
            Must provide `self.approx_vocab_size`. Generates the vocabulary based on
            the training data. To limit the number of samples the vocab generation
            looks at, override `self.max_samples_for_vocab`. Recommended and
            default.
          * `CHARACTER`: `ByteTextEncoder`, encode raw bytes.
          * `TOKEN`: `TokenTextEncoder`, vocabulary based on a file. Must provide a
            vocabulary file yourself (`TokenTextEncoder.store_to_file`) because one
            will not be generated for you. The vocab file should be stored in
            `data_dir/` with the name specified by `self.vocab_filename`.

        Returns:
          VocabType constant
        """
        return text_problems.VocabType.CHARACTER


    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        del dataset_split

        input_file = '/media/disk1/jennybae/data/t2t_data/tinyshakespeare/input.txt'

        with open(input_file, 'r') as f:
            data = f.read().strip().split('\n')
            data = [entity for entity in data if len(entity)>0]

        for line in data:
            if line:
                yield {
                    'inputs': line,
                    'targets': line[1:]
                }

@registry.register_model
class BasicCharRNN(t2t_model.T2TModel):
    def body(self, features):

        inputs = features.get('inputs')
        batch_size = common_layers.shape_list(inputs)[0]

        cells = []
        for i in range(self._hparams.num_hidden_layers):
            c = tf.nn.rnn_cell.LSTMCell(self._hparams.hidden_size)
            cells.append(c)

        cell = tf.nn.rnn_cell.MultiRNNCell(cells)
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)

        outputs, state = tf.nn.dynamic_rnn(cell, common_layers.flatten4d3d(inputs), initial_state = initial_state)


        return tf.expand_dims(outputs, axis=2)









#
# @registry.register_hparams
# def universal_transformer_attention_tiny():
#     hparams = universal_transformer_tiny()
#     hparams.add_hparam("expected_attention_loss_type", "softmax")
#     # Multiplier to the encoder-decoder expected attention loss.
#     hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
#     return hparams

# @registry.register_hparams
# def adaptive_universal_transformer_tiny_mine2():
#     hparams = universal_transformer_tiny()
#     hparams.recurrence_type = "act"
#     hparams.self_attention_type = "my_dot_product2"
#     return hparams
#
# @registry.register_hparams
# def adaptive_universal_transformer_tiny_mine():
#     hparams = universal_transformer_tiny()
#     hparams.recurrence_type = "act"
#     hparams.self_attention_type = "dot_product_mine"
#     return hparams

# @registry.register_hparams
# def lstm_seq2seq_attention_bidirectional():
#     """hparams for LSTM."""
#     hparams = common_hparams.basic_params1()
#     hparams.daisy_chain_variables = False
#     hparams.batch_size = 256
#     hparams.hidden_size = 128
#     hparams.num_hidden_layers = 1
#     hparams.initializer = "uniform_unit_scaling"
#     hparams.initializer_gain = 1.0
#     hparams.weight_decay = 0.0
#     return hparams
                   