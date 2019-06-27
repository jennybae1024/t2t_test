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
def transformer_sd0_l1_ld0_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld1_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld5_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld0_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld1_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld5_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld0_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld1_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld5_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld0_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld1_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld5_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld0_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld1_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld5_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld0_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld1_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld5_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld0_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld1_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld5_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld0_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld1_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld5_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld0_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld1_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd0_l1_ld5_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld0_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld1_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd0_l2_ld5_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.0
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld0_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld1_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld5_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld0_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld1_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld5_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld0_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld1_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld5_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld0_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld1_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld5_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld0_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld1_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld5_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld0_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld1_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld5_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld0_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld1_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld5_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld0_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld1_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld5_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld0_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld1_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd1_l1_ld5_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld0_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld1_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd1_l2_ld5_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.1
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld0_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld1_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld5_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld0_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld1_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld5_h25():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 25
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld0_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld1_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld5_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld0_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld1_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld5_h50():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 50
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld0_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld1_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld5_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld0_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld1_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld5_h100():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 100
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld0_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld1_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld5_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld0_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld1_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld5_h200():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 200
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld0_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld1_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd5_l1_ld5_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 1
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld0_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.0
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld1_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.1
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams


@registry.register_hparams
def transformer_sd5_l2_ld5_h400():
	hparams = transformer_tiny()
	hparams.symbol_dropout = 0.5
	hparams.num_hidden_layers = 2
	hparams.layer_prepostprocess_dropout = 0.5
	hparams.num_heads = 5
	hparams.hidden_size = 400
	return hparams

@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld0_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld1_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld5_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld0_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld1_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld5_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld0_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld1_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld5_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld0_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld1_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld5_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld0_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld1_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld5_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld0_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld1_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld5_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld0_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld1_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld5_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld0_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld1_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld5_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld0_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld1_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l1_ld5_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld0_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld1_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd0_l2_ld5_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.0
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld0_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld1_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld5_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld0_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld1_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld5_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld0_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld1_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld5_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld0_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld1_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld5_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld0_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld1_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld5_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld0_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld1_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld5_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld0_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld1_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld5_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld0_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld1_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld5_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld0_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld1_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l1_ld5_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld0_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld1_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd1_l2_ld5_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.1
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld0_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld1_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld5_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld0_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld1_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld5_h25():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 25
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld0_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld1_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld5_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld0_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld1_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld5_h50():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 50
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld0_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld1_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld5_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld0_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld1_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld5_h100():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 100
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld0_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld1_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld5_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld0_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld1_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld5_h200():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 200
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld0_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.0
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld1_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.1
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l1_ld5_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 1
    hparams.dropout = 0.5
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld0_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.0
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld1_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.1
    hparams.hidden_size = 400
    return hparams


@registry.register_hparams
def lstm_seq2seq_sd5_l2_ld5_h400():
    hparams = lstm_seq2seq()
    hparams.symbol_dropout = 0.5
    hparams.num_hidden_layers = 2
    hparams.dropout = 0.5
    hparams.hidden_size = 400
    return hparams