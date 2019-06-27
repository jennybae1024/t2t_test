from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
import numpy as np

import copy
from tensor2tensor.layers import common_hparams
from tensor2tensor.layers import common_layers, modalities, common_attention
from tensor2tensor.utils import registry, t2t_model
from tensor2tensor.models.transformer import transformer_encoder
from tensor2tensor.models.transformer import transformer_decoder
from tensor2tensor.models.transformer import transformer_encode
import tensorflow as tf


@registry.register_model
class LayerwiseTransformer(t2t_model.T2TModel):
    def __init__(self, *args, **kwargs):
        super(LayerwiseTransformer, self).__init__(*args, **kwargs)
        self.attention_weights = {}  # For visualizing attention heads.
        self.sym_noise = {}  # For visualizing attention heads.
        self.recurrent_memory_by_layer = None  # Override to enable recurrent memory
        self._encoder_function = transformer_encoder
        self._decoder_function = transformer_decoder

    def encode(self, inputs, target_space, hparams, features=None, losses=None):
        """Encode transformer inputs, see transformer_encode."""
        noise = hparams.get("sym_noise", None)
        if noise and hparams.mode == tf.estimator.ModeKeys.TRAIN :
            inputs = common_layers.scaling_noise(inputs, save_noise_to = self.sym_noise)
        return transformer_encode(
            self._encoder_function, inputs, target_space, hparams,
            attention_weights=self.attention_weights,
            features=features, losses=losses)

    def decode(self,
               decoder_input,
               encoder_output,
               encoder_decoder_attention_bias,
               decoder_self_attention_bias,
               hparams,
               cache=None,
               decode_loop_step=None,
               nonpadding=None,
               losses=None,
               **kwargs):
        """Decode Transformer outputs, see transformer_decode."""
        noise = hparams.get("sym_noise", None)
        if noise and hparams.mode == tf.estimator.ModeKeys.TRAIN :
            decoder_input = common_layers.scaling_noised(decoder_input, save_noise_to = self.sym_noise, shift = 1)
        res = tfm.transformer_decode(
            self._decoder_function, decoder_input, encoder_output,
            encoder_decoder_attention_bias, decoder_self_attention_bias,
            hparams, attention_weights=self.attention_weights, cache=cache,
            decode_loop_step=decode_loop_step, nonpadding=nonpadding, losses=losses,
            **kwargs)
        noise = hparams.get("sym_noise", None)
        if noise and hparams.mode == tf.estimator.ModeKeys.TRAIN :
            res = common_layers.scaling_noised(res, save_noise_to = self.sym_noise)
        return res


    def body(self, features):
        hparams = self._hparams

        losses = []

        if self.has_input:
            inputs = features["inputs"]
            target_space = features["target_space_id"]
            encoder_output, encoder_decoder_attention_bias = self.encode(
                inputs, target_space, hparams, features=features, losses=losses)
        else:
            encoder_output, encoder_decoder_attention_bias = (None, None)

        targets = features["targets"]
        targets_shape = common_layers.shape_list(targets)
        targets = common_layers.flatten4d3d(targets)
        decoder_input, decoder_self_attention_bias = tfm.transformer_prepare_decoder(
            targets, hparams, features=features)

        decode_kwargs = {}
        if self.recurrent_memory_by_layer is not None:
            chunk_number_each_token = tf.squeeze(features["chunk_number"], (-1, -2))
            chunk_number_each_batch = chunk_number_each_token[:, 0]
            decode_kwargs = dict(
                recurrent_memory_by_layer=self.recurrent_memory_by_layer,
                chunk_number=chunk_number_each_batch,
            )

        decoder_output = self.decode(
            decoder_input,
            encoder_output,
            encoder_decoder_attention_bias,
            decoder_self_attention_bias,
            hparams,
            nonpadding=tfm.features_to_nonpadding(features, "targets"),
            losses=losses,
            **decode_kwargs
        )

        expected_attention_loss_type = hparams.get("expected_attention_loss_type")
        if expected_attention_loss_type is not None:
            expected_attentions = features.get("expected_attentions")
            if expected_attentions is not None:
                attention_loss = common_attention.encoder_decoder_attention_loss(
                    expected_attentions, self.attention_weights,
                    hparams.expected_attention_loss_type,
                    hparams.expected_attention_loss_multiplier)

            else:
                attention_loss = 0

            expected_enc_attentions_loss_type = hparams.get("expected_enc_attention_loss_type")
            if expected_enc_attentions_loss_type is not None:
                expected_enc_attentions = features.get("expected_enc_attentions")
                if expected_attentions is not None:
                    enc_attention_loss = common_attention.encoder_encoder_attention_loss(
                        expected_enc_attentions, self.attention_weights,
                        hparams.expected_enc_attention_loss_type,
                        hparams.expected_enc_attention_loss_multiplier)
                    return decoder_output, {"attention_loss": attention_loss, "enc_attention_loss": enc_attention_loss}

            return decoder_output, {"attention_loss": attention_loss}

        ret = tf.reshape(decoder_output, targets_shape)
        if losses:
            return ret, {"extra_loss": tf.add_n(losses)}
        else:
            return ret

    def _greedy_infer(self, features, decode_length, use_tpu=False):
        # For real-valued modalities use the slow decode path for now.
        if (self._target_modality_is_real or
                self._hparams.self_attention_type != "dot_product"):
            return super(LayerwiseTransformer, self)._greedy_infer(features, decode_length)
        with tf.variable_scope(self.name):
            return self._fast_decode(features, decode_length)

    def _fast_decode(self,
                     features,
                     decode_length,
                     beam_size=1,
                     top_beams=1,
                     alpha=1.0):
        if self._num_datashards != 1:
            raise NotImplementedError("Fast decoding only supports a single shard.")
        dp = self._data_parallelism
        hparams = self._hparams
        target_modality = self._problem_hparams.modality["targets"]
        target_vocab_size = self._problem_hparams.vocab_size["targets"]
        if target_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
            target_vocab_size += (-target_vocab_size) % hparams.vocab_divisor
        if "targets_segmentation" in features:
            raise NotImplementedError(
                "Decoding not supported on packed datasets "
                " If you want to decode from a dataset, use the non-packed version"
                " of the dataset when decoding.")
        if self.has_input:
            inputs = features["inputs"]
            if target_modality == modalities.ModalityType.CLASS_LABEL:
                decode_length = 1
            else:
                decode_length = (
                        common_layers.shape_list(inputs)[1] + features.get(
                    "decode_length", decode_length))
            # TODO(llion): Clean up this reshaping logic.
            inputs = tf.expand_dims(inputs, axis=1)
            if len(inputs.shape) < 5:
                inputs = tf.expand_dims(inputs, axis=4)
            s = common_layers.shape_list(inputs)
            batch_size = s[0]
            inputs = tf.reshape(inputs, [s[0] * s[1], s[2], s[3], s[4]])
            # _shard_features called to ensure that the variable names match
            inputs = self._shard_features({"inputs": inputs})["inputs"]
            input_modality = self._problem_hparams.modality["inputs"]
            input_vocab_size = self._problem_hparams.vocab_size["inputs"]
            if input_vocab_size is not None and hasattr(hparams, "vocab_divisor"):
                input_vocab_size += (-input_vocab_size) % hparams.vocab_divisor
            modality_name = hparams.name.get(
                "inputs",
                modalities.get_name(input_modality))(hparams, input_vocab_size)
            with tf.variable_scope(modality_name):
                bottom = hparams.bottom.get("inputs",
                                            modalities.get_bottom(input_modality))
                inputs = dp(bottom, inputs, hparams, input_vocab_size)
            with tf.variable_scope("body"):
                encoder_output, encoder_decoder_attention_bias = dp(
                    self.encode,
                    inputs,
                    features["target_space_id"],
                    hparams,
                    features=features)
            encoder_output = encoder_output[0]
            encoder_decoder_attention_bias = encoder_decoder_attention_bias[0]
            partial_targets = None
        else:
            # The problem has no inputs.
            encoder_output = None
            encoder_decoder_attention_bias = None

            # Prepare partial targets.
            # In either features["inputs"] or features["targets"].
            # We force the outputs to begin with these sequences.
            partial_targets = features.get("inputs")
            if partial_targets is None:
                partial_targets = features["targets"]
            assert partial_targets is not None
            partial_targets = common_layers.expand_squeeze_to_nd(partial_targets, 2)
            partial_targets = tf.to_int64(partial_targets)
            partial_targets_shape = common_layers.shape_list(partial_targets)
            partial_targets_length = partial_targets_shape[1]
            decode_length = (
                    partial_targets_length + features.get("decode_length", decode_length))
            batch_size = partial_targets_shape[0]

        if hparams.pos == "timing":
            positional_encoding = common_attention.get_timing_signal_1d(
                decode_length + 1, hparams.hidden_size)
        elif hparams.pos == "emb":
            positional_encoding = common_attention.add_positional_embedding(
                tf.zeros([1, decode_length, hparams.hidden_size]), hparams.max_length,
                "body/targets_positional_embedding", None)
        else:
            positional_encoding = None

        def preprocess_targets(targets, i):
            """Performs preprocessing steps on the targets to prepare for the decoder.

            This includes:
              - Embedding the ids.
              - Flattening to 3D tensor.
              - Optionally adding timing signals.

            Args:
              targets: inputs ids to the decoder. [batch_size, 1]
              i: scalar, Step number of the decoding loop.

            Returns:
              Processed targets [batch_size, 1, hidden_dim]
            """
            # _shard_features called to ensure that the variable names match
            targets = self._shard_features({"targets": targets})["targets"]
            modality_name = hparams.name.get(
                "targets",
                modalities.get_name(target_modality))(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                bottom = hparams.bottom.get(
                    "targets", modalities.get_targets_bottom(target_modality))
                targets = dp(bottom, targets, hparams, target_vocab_size)[0]
            targets = common_layers.flatten4d3d(targets)

            # GO embeddings are all zero, this is because transformer_prepare_decoder
            # Shifts the targets along by one for the input which pads with zeros.
            # If the modality already maps GO to the zero embeddings this is not
            # needed.
            targets = tf.cond(
                tf.equal(i, 0), lambda: tf.zeros_like(targets), lambda: targets)

            if positional_encoding is not None:
                targets += positional_encoding[:, i:i + 1]
            return targets

        decoder_self_attention_bias = (
            common_attention.attention_bias_lower_triangle(decode_length))
        if hparams.proximity_bias:
            decoder_self_attention_bias += common_attention.attention_bias_proximal(
                decode_length)

        def symbols_to_logits_fn(ids, i, cache):
            """Go from ids to logits for next symbol."""
            ids = ids[:, -1:]
            targets = tf.expand_dims(tf.expand_dims(ids, axis=2), axis=3)
            targets = preprocess_targets(targets, i)

            bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]

            with tf.variable_scope("body"):
                body_outputs = dp(
                    self.decode,
                    targets,
                    cache.get("encoder_output"),
                    cache.get("encoder_decoder_attention_bias"),
                    bias,
                    hparams,
                    cache,
                    nonpadding=tfm.features_to_nonpadding(features, "targets"))

            modality_name = hparams.name.get(
                "targets",
                modalities.get_name(target_modality))(hparams, target_vocab_size)
            with tf.variable_scope(modality_name):
                top = hparams.top.get("targets", modalities.get_top(target_modality))
                logits = dp(top, body_outputs, None, hparams, target_vocab_size)[0]

            ret = tf.squeeze(logits, axis=[1, 2, 3])
            if partial_targets is not None:
                # If the position is within the given partial targets, we alter the
                # logits to always return those values.
                # A faster approach would be to process the partial targets in one
                # iteration in order to fill the corresponding parts of the cache.
                # This would require broader changes, though.
                vocab_size = tf.shape(ret)[1]

                def forced_logits():
                    return tf.one_hot(
                        tf.tile(partial_targets[:, i], [beam_size]), vocab_size, 0.0,
                        -1e9)

                ret = tf.cond(
                    tf.less(i, partial_targets_length), forced_logits, lambda: ret)
            return ret, cache

        ret = tfm.fast_decode(
            encoder_output=encoder_output,
            encoder_decoder_attention_bias=encoder_decoder_attention_bias,
            symbols_to_logits_fn=symbols_to_logits_fn,
            hparams=hparams,
            decode_length=decode_length,
            vocab_size=target_vocab_size,
            beam_size=beam_size,
            top_beams=top_beams,
            alpha=alpha,
            batch_size=batch_size,
            force_decode_length=self._decode_hparams.force_decode_length)
        if partial_targets is not None:
            if beam_size <= 1 or top_beams <= 1:
                ret["outputs"] = ret["outputs"][:, partial_targets_length:]
            else:
                ret["outputs"] = ret["outputs"][:, :, partial_targets_length:]
        return ret




@registry.register_hparams
def layerwise_transformer_tiny():
    hparams = transformer_tiny()
    return hparams


