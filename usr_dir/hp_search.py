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
from tensor2tensor.models.lstm import lstm_seq2seq
from tensor2tensor.models.image_transformer import imagetransformer_base
from tensor2tensor.models.xception import xception_base
from tensor2tensor.models.lstm import lstm
from tensor2tensor.models.research.universal_transformer import universal_transformer_tiny
from tensor2tensor.models.research.universal_transformer import update_hparams_for_universal_transformer
import tensorflow as tf



@registry.register_hparams
def transformer_l1_h25():
    hparams = transformer_tiny()
    hparams.hidden_size = 25
    hparams.num_heads = 5
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def transformer_l1_h50():
    hparams = transformer_tiny()
    hparams.hidden_size = 50
    hparams.num_heads = 5
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def transformer_l1_h100():
    hparams = transformer_tiny()
    hparams.hidden_size = 100
    hparams.num_heads = 5
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def transformer_l1_h200():
    hparams = transformer_tiny()
    hparams.hidden_size = 200
    hparams.num_heads = 5
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def transformer_l1_h400():
    hparams = transformer_tiny()
    hparams.hidden_size = 400
    hparams.num_heads = 5
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def transformer_l2_h25():
    hparams = transformer_l1_h25()
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def transformer_l2_h50():
    hparams = transformer_l1_h50()
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def transformer_l2_h100():
    hparams = transformer_l1_h100()
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def transformer_l2_h200():
    hparams = transformer_l1_h200()
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def transformer_l2_h400():
    hparams = transformer_l1_h400()
    hparams.num_hidden_layers = 2
    return hparams



@registry.register_hparams
def lstm_seq2seq_l2_h25():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 25
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def lstm_seq2seq_l2_h50():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 50
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def lstm_seq2seq_l2_h100():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 100
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def lstm_seq2seq_l2_h200():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 200
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def lstm_seq2seq_l2_h400():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 400
    hparams.num_hidden_layers = 2
    return hparams

@registry.register_hparams
def lstm_seq2seq_l1_h25():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 25
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def lstm_seq2seq_l1_h50():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 50
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def lstm_seq2seq_l1_h100():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 100
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def lstm_seq2seq_l1_h200():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 200
    hparams.num_hidden_layers = 1
    return hparams

@registry.register_hparams
def lstm_seq2seq_l1_h400():
    hparams = lstm_seq2seq()
    hparams.hidden_size = 400
    hparams.num_hidden_layers = 1
    return hparams




@registry.register_hparams
def transformer_l1_tiny():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 1
    return hparams


@registry.register_hparams
def transformer_position_random_tiny():
    hparams = transformer_tiny()
    hparams.position_start_index = "random"
    return hparams


@registry.register_hparams
def transformer_single_head_tiny():
    hparams = transformer_tiny()
    hparams.num_heads = 1
    return hparams

@registry.register_hparams
def transformer_normalize_enc_single_head_tiny():
    hparams = transformer_tiny()
    hparams.num_heads = 1
    hparams.add_hparam("enc_self_attention_type", 'dot_product_normalize_enc')
    return hparams

@registry.register_hparams
def transformer_single_head_cnn_enc_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("enc_self_attention_type", 'dot_product_cnn1')
    hparams.num_heads = 1
    return hparams


@registry.register_hparams
def transformer_l3_tiny():
    hparams = transformer_tiny()
    hparams.num_hidden_layers = 3
    return hparams

@registry.register_hparams
def transformer_position_random_l3_tiny():
    hparams = transformer_tiny()
    hparams.position_start_index = "random"
    hparams.num_hidden_layers = 3
    return hparams

@registry.register_hparams
def transformer_l1_h64_tiny():
    hparams = transformer_tiny()
    hparams.hidden_layers = 1
    hparams.hidden_size = 64
    return hparams


@registry.register_hparams
def transformer_l2_h64_tiny():
    hparams = transformer_tiny()
    hparams.hidden_size = 64
    return hparams


@registry.register_hparams
def transformer_l3_h64_tiny():
    hparams = transformer_tiny()
    hparams.hidden_layers = 3
    hparams.hidden_size = 64
    return hparams

@registry.register_hparams
def transformer_l4_h64_tiny():
    hparams = transformer_tiny()
    hparams.hidden_layers = 4
    hparams.hidden_size = 64
    return hparams

@registry.register_hparams
def transformer_no_pos_tiny():
    hparams = transformer_tiny()
    hparams.pos = None
    return hparams

@registry.register_hparams
def transformer_no_pos_cnn_enc_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("enc_self_attention_type", 'dot_product_cnn')
    hparams.pos = None
    return hparams


@registry.register_hparams
def xception_small():
    hparams = xception_base()
    hparams.batch_size = 128
    hparams.hidden_size = 64
    hparams.num_hidden_layers = 2
    hparams.learning_rate_decay_scheme = "none"
    return hparams


@registry.register_hparams
def imagetransformer_mnist_tiny():
    hparams = imagetransformer_base()
    hparams.num_decoder_layers = 2
    hparams.hidden_size = 64
    hparams.batch_size = 1
    hparams.unconditional = True
    hparams.num_channels = 1
    hparams.max_length = 66000  # allow for 256x256
    return hparams

