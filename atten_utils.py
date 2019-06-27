import tensorflow as tf
import numpy as np
import os
from tensor2tensor.utils import trainer_lib, registry, hparams_lib
from tensor2tensor.layers import common_hparams
from tensor2tensor.data_generators import text_encoder
import re
import pandas as pd
import itertools


SIZE = 60
DATA_DIR = "/media/disk1/ksk5693/data/t2t_data/"
SESS_DIR = "/media/disk1/ksk5693/sessions/t2t/"

def path_set_up(model_name, hparam_set, problem_name, global_steps=None):
    data_dir = os.path.expanduser(DATA_DIR)
    # tmp_dir = os.path.expanduser("~/t2t/tmp")
    train_dir = os.path.expanduser(SESS_DIR+problem_name+'-'+model_name+'-'+hparam_set)
    ckp = train_dir+"/model.ckpt-"+str(global_steps) if global_steps else train_dir
    return data_dir, train_dir, ckp


def hparams_set_up(problem_name, hparam_set=None, hparams_override=None):
    if hparam_set:
        hparams = trainer_lib.create_hparams(hparam_set, \
                                             hparams_overrides_str=hparams_override, \
                                             )
    else:
        hparams = common_hparams.basic_params1()
    hparams.data_dir = DATA_DIR
    hparams_lib.add_problem_hparams(hparams, problem_name)
    return hparams, hparams.problem


def model_hp_set_up(problem_name, basic_dir=SESS_DIR):
    model_hp = {}
    for i in os.listdir(basic_dir):
        if problem_name in i:
            de = [idx for idx, j in enumerate(i) if j == '-']
            model_name = i[de[0]+1:de[1]]
            hp_name = i[de[1]+1:]
            if model_hp.get(model_name, -1) == -1:
                model_hp[model_name] = [hp_name]
            else:
                model_hp[model_name].append(hp_name)
    return model_hp

# Setup helper functions for encoding and decoding
def instant_encode(encoders, input_str, append_eos = True):
    """Input str to features dict, ready for inference"""
    if len(re.findall("[0-9]", input_str)) == 0:
        inputs = encoders["inputs"].encode(input_str)
        if append_eos:
            inputs = inputs + [1]
    else:
        inputs = [encoders["inputs"].encode(i)[0] for i in input_str]
        if append_eos:
            inputs = inputs + [1]
    batch_inputs = tf.reshape(inputs, [1, -1, 1])  # Make it 3D.
    return {"inputs": batch_inputs}


def instant_decode(encoders, integers):
    """List of ints to str"""
    integers = list(np.squeeze(integers))
    if 1 in integers:
        integers = integers[:integers.index(1)]
    return encoders["inputs"].decode(np.squeeze(integers))


def encode_eval(encoders, input_str, output_str):
    if len(re.findall("[0-9]", input_str)) == 0:
        inputs = tf.reshape(encoders["inputs"].encode(input_str) + [1], [1, -1, 1, 1])  # Make it 3D.
        outputs = tf.reshape(encoders["inputs"].encode(output_str) + [1], [1, -1, 1, 1])  # Make it 3D.
    else:
        inputs = tf.reshape([encoders["inputs"].encode(i)[0] for i in input_str] + [1], [1, -1, 1, 1])  # Make it 3D.
        outputs = tf.reshape([encoders["inputs"].encode(i)[0] for i in output_str] + [1], [1, -1, 1, 1])  # Make it 3D.
    return {"inputs": inputs, "targets": outputs}


def att_names(translate_model):
    names = []
    for i in translate_model.attention_weights.keys():
        names.append(re.sub("[0-9]", "%d", i))
    return list(set(names))

def get_att_mats(translate_model, enc_att_name, dec_att_name, encdec_att_name, resizing=False):
    enc_atts = []
    dec_atts = []
    encdec_atts = []

    for i in range(translate_model.hparams.num_hidden_layers):
        enc_att = translate_model.attention_weights[enc_att_name % i][0]
        dec_att = translate_model.attention_weights[dec_att_name % i][0]
        encdec_att = translate_model.attention_weights[encdec_att_name % i][0]
        if resizing:
            enc_atts.append(resize(enc_att))
            dec_atts.append(resize(dec_att))
            encdec_atts.append(resize(encdec_att))
        else:
            enc_atts.append(enc_att)
            dec_atts.append(dec_att)
            encdec_atts.append(encdec_att)
    return enc_atts, dec_atts, encdec_atts

def resize(np_mat):
    # Sum across heads
    np_mat = np_mat[:, :SIZE, :SIZE]
    row_sums = np.sum(np_mat, axis=0)
    # Normalize
    layer_mat = np_mat / row_sums[np.newaxis, :]
    lsh = layer_mat.shape
    # Add extra dim for viz code to work.
    layer_mat = np.reshape(layer_mat, (1, lsh[0], lsh[1], lsh[2]))
    return layer_mat

def to_tokens(ids, hparams):
    ids = np.squeeze(ids)
    subtokenizer = hparams.problem_hparams.vocabulary['targets']
    tokens = []
    for _id in ids:
        if _id == 0:
            tokens.append('<PAD>')
        elif _id == 1:
            tokens.append('<EOS>')
        elif _id == -1:
            tokens.append('<NULL>')
        else:
            tokens.append(subtokenizer.decode([_id]))
    return tokens

def call_html():
    import IPython
    display(IPython.core.display.HTML('''
        <script src="/static/components/requirejs/require.js"></script>
        <script>
          requirejs.config({
            paths: {
              base: '/static/base',
              "d3": "https://cdnjs.cloudflare.com/ajax/libs/d3/3.5.8/d3.min",
              jquery: '//ajax.googleapis.com/ajax/libs/jquery/2.0.0/jquery.min',
            },
          });
        </script>
        '''))
    
    
    
def score_file(model_name, problem_name, hparam_set, global_steps, shard, file="test"):
    """Score each line in a file and return the scores."""
        
    data_dir, train_dir, ckp = path_set_up(model_name, hparam_set, problem_name, global_steps)
    # hparams, problem = hparams_set_up(problem_name, hparam_set)
    decode_dir = os.path.join(train_dir, "decode_00000")
    for filename_ in os.listdir(decode_dir):
        if len(re.findall("decodes", filename_)) > 0 and len(re.findall(str(global_steps)+file+str(shard), filename_)):
            decodes_filename = filename_
        if len(re.findall("inputs", filename_)) > 0 and len(re.findall(str(global_steps)+file+str(shard), filename_)):
            inputs_filename = filename_
        if len(re.findall("targets", filename_)) > 0 and len(re.findall(str(global_steps)+file+str(shard), filename_)):
            targets_filename = filename_

    with tf.gfile.Open(os.path.join(decode_dir, decodes_filename)) as f:
        lines = f.readlines()
        output_lines = [line.strip() for line in lines]
    with tf.gfile.Open(os.path.join(decode_dir, inputs_filename)) as f:
        lines = f.readlines()
        input_lines = [line.strip() for line in lines]
    with tf.gfile.Open(os.path.join(decode_dir, targets_filename)) as f:
        lines = f.readlines()
        target_lines = [line.strip() for line in lines]

    return input_lines, output_lines, target_lines


def acc_per_seq(output_lines, target_lines, tokens=False):
    res = 0.0
    num = len(target_lines)
    if tokens:
        for output_line, target_line in zip(output_lines, target_lines):
            output_tokens = output_line.split()
            target_tokens = target_line.split()
            try:
                if target_tokens == output_tokens[:len(target_tokens)]:
                    res += 1.0
            except:
                continue
        res = res / num
        return res

    for output_line, target_line in zip(output_lines, target_lines):
        if output_line == target_line:
            res += 1.0
    res = res / num
    return res


def acc_per_char(output_lines, target_lines, reverse=False, tokens=False):
    res = 0.0
    if tokens:
        if reverse:
            num = sum([len(output_line.split()) for output_line in output_lines])
            for output_line, target_line in zip(output_lines, target_lines):
                for idx, target_sym in enumerate(target_line.split()):
                    try:
                        if target_sym == output_line.split()[idx]:
                            res += 1.0
                    except:
                        continue
            res = res / num if num > 0 else 0
        else:
            num = sum([len(target_line.split()) for target_line in target_lines])
            for output_line, target_line in zip(output_lines, target_lines):
                for idx, output_sym in enumerate(output_line.split()):
                    try:
                        if output_sym == target_line.split()[idx]:
                            res += 1.0
                    except:
                        continue
            res = res / num if num > 0 else 0
        return res

    if reverse:
        num = sum([len(output_line) for output_line in output_lines])
        for output_line, target_line in zip(output_lines, target_lines):
            for idx, target_sym in enumerate(target_line):
                try:
                    if target_sym == output_line[idx]:
                        res += 1.0
                except:
                    continue
        res = res / num if num > 0 else 0
    else:
        num = sum([len(target_line) for target_line in target_lines])
        for output_line, target_line in zip(output_lines, target_lines):
            for idx, output_sym in enumerate(output_line):
                try:
                    if output_sym == target_line[idx]:
                        res += 1.0
                except:
                    continue
        res = res / num if num > 0 else 0
    return res

def summary(problem_name, model_hp, global_steps_set, shard_set, file="test", tokens=False):
    column_names = ["model", "steps", "shard", "avg_len", "seq_acc", "seq_rev_acc", "char_acc", "char_acc_reverse"]
    df = pd.DataFrame(columns=column_names)
    for model_name, hparam_set in model_hp.items():
        for hparam_name, global_steps, shard in itertools.product(hparam_set, global_steps_set, shard_set):
            try:
                input_lines, output_lines, target_lines = score_file(model_name, \
                                                                     problem_name, \
                                                                     hparam_name, \
                                                                     global_steps, \
                                                                     shard, file=file)
                a = acc_per_seq(output_lines, target_lines, tokens=tokens)
                aa = acc_per_seq(target_lines, output_lines, tokens=tokens)
                b = acc_per_char(output_lines, target_lines, tokens, tokens=tokens)
                c = acc_per_char(output_lines, target_lines, reverse=True, tokens=tokens)
                ratio = np.average([len(oline) / len(tline) for oline, tline in zip(output_lines, target_lines)])

                res = [hparam_name, global_steps, shard, ratio, a, aa, b, c]
                df = df.append([dict(zip(column_names, res))], ignore_index=True)
            except:
                print("NO DECODED FILE " + hparam_name + " / " + str(global_steps)+ " / " + str(shard))
    return df

    # Prepare features for feeding into the model.
    # inputs_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    # batch_inputs = tf.reshape(inputs_ph, [1, -1, 1, 1])  # Make it 4D.
    #
    # targets_ph = tf.placeholder(dtype=tf.int32)  # Just length dimension.
    # batch_targets = tf.reshape(targets_ph, [1, -1, 1, 1])  # Make it 4D.
    #
    # for line in output_lines:
    #     tab_split = line.split("\t")
    #     targets = tab_split[1].strip()
    #     inputs = tab_split[0].strip()
    #     # Run encoders and append EOS symbol.
    #     targets_numpy = encoders["targets"].encode(targets) + [text_encoder.EOS_ID]
    #     if has_inputs:
    #         inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
    #         # Prepare the feed.
    #         feed = {inputs_ph: inputs_numpy, targets_ph: targets_numpy}
    #
    # features = {"inputs": batch_inputs, "targets": batch_targets}

    # Prepare the model and the graph when model runs on features.
    # model = registry.model(model_name)(hparams, tf.estimator.ModeKeys.EVAL)
    # _, losses = model(features)
    # saver = tf.train.Saver()
    #
    # with tf.Session() as sess:
    #     # Load weights from checkpoint.
    #     saver.restore(sess, ckp)
    #     # Run on each line.
    #     with tf.gfile.Open(os.path.join(decode_dir,filename)) as f:
    #         lines = f.readlines()
    #         results = []
    #     for line in lines:
    #         tab_split = line.split("\t")
    #         if len(tab_split) > 2:
    #             raise ValueError("Each line must have at most one tab separator.")
    #         if len(tab_split) == 1:
    #             targets = tab_split[0].strip()
    #         else:
    #             targets = tab_split[1].strip()
    #             inputs = tab_split[0].strip()
    #         # Run encoders and append EOS symbol.
    #         targets_numpy = encoders["targets"].encode(targets) + [text_encoder.EOS_ID]
    #         if has_inputs:
    #             inputs_numpy = encoders["inputs"].encode(inputs) + [text_encoder.EOS_ID]
    #             # Prepare the feed.
    #             feed = {inputs_ph: inputs_numpy, targets_ph: targets_numpy}
    #         else:
    #             feed = {targets_ph: targets_numpy}
    #         # Get the score.
    #         np_loss = sess.run(losses["training"], feed)
    #         results.append(np_loss)
    # return results

