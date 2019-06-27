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

@registry.register_model
class GRUSeq2seqAttentionBidirectionalEncoder(t2t_model.T2TModel):
    """Seq to seq LSTM with attention."""

    def body(self, features):
        # TODO(lukaszkaiser): investigate this issue and repair.
        if self._hparams.initializer == "orthogonal":
            raise ValueError("LSTM models fail with orthogonal initializer.")
        train = self._hparams.mode == tf.estimator.ModeKeys.TRAIN
        
        inputs = features.get("inputs")
        targets = features["targets"]
        hparams = self._hparams

        with tf.variable_scope("gru_seq2seq_attention_bid_encoder"):
            inputs_length = common_layers.length_from_embedding(inputs)
            # gru encoder.
            encoder_outputs, final_encoder_state = gru_bid_encoder(inputs, inputs_length, hparams, train, "encoder")
            # LSTM decoder with attention
            shifted_targets = common_layers.shift_right(targets)
            # Add 1 to account for the padding added to the left from shift_right
            targets_length = common_layers.length_from_embedding(shifted_targets) + 1
            hparams_decoder = copy.copy(hparams)
            hparams_decoder.hidden_size = hparams.hidden_size
            decoder_outputs = gru_attention_decoder(
                common_layers.flatten4d3d(shifted_targets), hparams_decoder, train,
                "decoder", final_encoder_state, encoder_outputs,
                inputs_length, targets_length)
            return tf.expand_dims(decoder_outputs, axis=2)    
            
            

def _dropout_gru_cell(hparams, train, multi=1):
    return tf.nn.rnn_cell.DropoutWrapper(
        tf.nn.rnn_cell.GRUCell(hparams.hidden_size/multi),
        input_keep_prob=1.0 - hparams.dropout * tf.to_float(train))


def gru_bid_encoder(inputs, sequence_length, hparams, train, name):
    """Bidirectional GRU for encoding inputs that are [batch x time x size]."""
    with tf.variable_scope(name):
        cell_fw = tf.nn.rnn_cell.MultiRNNCell(
            [_dropout_gru_cell(hparams, train, 2)
             for _ in range(hparams.num_hidden_layers)])

        cell_bw = tf.nn.rnn_cell.MultiRNNCell(
            [_dropout_gru_cell(hparams, train, 2)
             for _ in range(hparams.num_hidden_layers)])
        encoder_outputs, encoder_states = tf.nn.bidirectional_dynamic_rn
            cell_fw,
            cell_bw,
            tf.squeeze(inputs, axis=2),
            sequence_length,
            dtype=tf.float32,
            time_major=False)
        
        if isinstance(encoder_outputs, tuple):
            encoder_outputs = tf.concat(encoder_outputs, 2)
        return encoder_outputs, encoder_states       
            
def gru_attention_decoder(inputs, hparams, train, name, initial_state,
                           encoder_outputs, encoder_output_length,
                           decoder_input_length):
    layers = [_dropout_gru_cell(hparams, train)
              for _ in range(hparams.num_hidden_layers)]
    
    atten = attn_head_vanila(hparams.hidden_size)
    

    batch_size = common_layers.shape_list(inputs)[0]

    initial_st = []
    for i in range(hparams.num_hidden_layers):
        # GRU
        initial_st.append(tf.concat([initial_state[0][i], initial_state[1][i]], axis=-1))            
    initial_st=tuple(initial_st) 
    cell = tf.nn.rnn_cell.MultiRNNCell(layers)
    
    with tf.variable_scope(name):
        decoder_outputs, _ = tf.nn.dynamic_rnn(cell, inputs, decoder_input_length, initial_state=initial_st,
                                        dtype=tf.float32, time_major=False)        
        
        output = atten(decoder_outputs, encoder_outputs, encoder_outputs)
        
        
        # output is [batch_size, decoder_steps, attention_size], where
        # attention_size is either hparams.hidden_size (when
        # hparams.output_attention is 0) or hparams.attention_layer_size (when
        # hparams.output_attention is 1) times the number of attention heads.
        #
        # For multi-head attention project output back to hidden size.
        if hparams.output_attention == 1 and hparams.num_heads > 1:
            output = tf.layers.dense(output, hparams.hidden_size)
        return output

class attn_head_vanila(object):
    def __init__(self, hidden_dim, scope=None):
        self.scope = scope if scope != None else tf.get_variable_scope()
        self.hidden_dim = hidden_dim
        
    def batch_lin(self, input, weight, bias):
        b=tf.expand_dims(tf.expand_dims(bias, axis=0), axis=0)
        batch_size, seq_len  = tf.shape(input)[0], tf.shape(input)[1]
#         print(input)
#         print(weight)
#         print(b)
        return tf.nn.tanh(tf.einsum('bqv, vo->bqo', input, weight)) + tf.tile(b, [batch_size, seq_len, 1])
    
    def __call__(self, Q, K, V):
        with tf.variable_scope(self.scope, reuse=tf.AUTO_REUSE):
#             print(self.hidden_dim)
            W = tf.get_variable("W", shape=[self.hidden_dim*2, self.hidden_dim])
            b = tf.get_variable("b", shape=[self.hidden_dim])
            context = scaled_dot_product_attn_vanila(Q, K, V)
            combined = tf.concat([Q, context], axis=-1)
            return tf.identity(self.batch_lin(combined, W, b), name = "eval")
        
def scaled_dot_product_attn_vanila(Q, K, V):
    # Q : batch_size x num_of_queries x d_k = bqk
    # K : batch_size x size_of_dic x d_k = bdk
    # V : batch_size x size_of_dic x d_v = bdv
    # output : batch_size x num_of_queries x d_v = bqv 
    temp = tf.einsum('bqk,bdk->bqd', Q, K)
    atten = tf.nn.softmax(temp, axis=-1, name="alphas")
    return tf.einsum('bqd,bdv->bqv', atten, V)


        
# @registry.register_hparams
# def my_transformer():
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

# @registry.register_hparams
# def universal_transformer_tiny_mine():
#     hparams = transformer_tiny()
#     hparams = update_hparams_for_universal_transformer(hparams)
#     hparams.num_rec_steps = 8
#     hparams.self_attention_type = "dot_product_mine"
#     return hparams

# @registry.register_hparams
# def transformer_tiny_mine():
#     hparams = transformer_base()
#     hparams.num_hidden_layers = 2
#     hparams.hidden_size = 128
#     hparams.filter_size = 512
#     hparams.self_attention_type = "dot_product_mine"
#     hparams.num_heads = 4
#     return hparams



@registry.register_hparams
def transformer_sym_noise_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("sym_noise", True)
    return hparams

@registry.register_hparams
def transformer_sym_noise_l2_tiny():
    hparams = transformer_tiny()
    hparams.loss = {'targets': modalities.generic_sym_l2_loss}
    hparams.add_hparam("sym_noise", True)
    return hparams

@registry.register_hparams
def transformer_cnn_enc_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("enc_self_attention_type", 'dot_product_cnn')
    return hparams

@registry.register_hparams
def transformer_cnn_prob_enc_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("enc_self_attention_type", 'dot_product_prob_cnn')
    return hparams

@registry.register_hparams
def transformer_cnn_prob_tiny():
    hparams = transformer_tiny()
    hparams.self_attention_type = 'dot_product_prob_cnn'
    return hparams

@registry.register_hparams
def transformer_l2_tiny():
    hparams = transformer_tiny()
    hparams.loss = {'targets': modalities.generic_sym_l2_loss}
    return hparams

@registry.register_hparams
def transformer_attention2_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # Multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 10.0)
    return hparams


@registry.register_hparams
def transformer_position_random_timing_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("expected_attention_loss_type", "softmax")
    hparams.position_start_index = "random"
    # Multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams


@registry.register_hparams
def transformer_enc_dec_supervision_position_random_timing_tiny():
    hparams = transformer_tiny()
    hparams.position_start_index = "random"
    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # Multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams

@registry.register_hparams
def transformer_both_supervision_position_random_timing_tiny():
    hparams = transformer_tiny()
    hparams.position_start_index = "random"
    hparams.add_hparam("expected_enc_attention_loss_type", "softmax")
    hparams.add_hparam("expected_enc_attention_loss_multiplier", 1.0)

    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams

@registry.register_hparams
def transformer_sym_noise_enc_dec_supervision_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("sym_noise", True)
    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # Multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams

@registry.register_hparams
def transformer_sym_noise_both_supervision_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("sym_noise", True)
    hparams.add_hparam("expected_enc_attention_loss_type", "softmax")
    hparams.add_hparam("expected_enc_attention_loss_multiplier", 1.0)
    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # Multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams

@registry.register_hparams
def transformer_enc_dec_supervision_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # Multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams

@registry.register_hparams
def transformer_both_supervision_tiny():
    hparams = transformer_tiny()
    hparams.add_hparam("expected_enc_attention_loss_type", "softmax")
    hparams.add_hparam("expected_enc_attention_loss_multiplier", 1.0)

    hparams.add_hparam("expected_attention_loss_type", "softmax")
    # multiplier to the encoder-decoder expected attention loss.
    hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
    return hparams

# @registry.register_hparams
# def adaptive_universal_transformer_position_random_timing_attention_tiny():
#     """HParams for supervised attention problems."""
#     hparams = universal_transformer_tiny()
#     hparams.recurrence_type = "act"
#     hparams.position_start_index = "random"
#     # Attention loss type (KL-divergence or MSE).
#     hparams.add_hparam("expected_attention_loss_type", "softmax")
#     # Multiplier to the encoder-decoder expected attention loss.
#     hparams.add_hparam("expected_attention_loss_multiplier", 1.0)
#     return hparams
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
                   