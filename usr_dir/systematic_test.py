import re
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import numpy as np
import random
import string
import tensorflow as tf

class AlgorithmicString(text_problems.Text2TextProblem):
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
class AlgorithmicUnseenIdentity(AlgorithmicString):
    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "expected_attentions": tf.VarLenFeature(tf.float32),
            "expected_enc_attentions": tf.VarLenFeature(tf.float32)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def generate_expected_attention(self, enc):
        num = len(enc) + 1
        return [float(i) for i in range(num)]

    def generate_expected_enc_attention(self, enc):
        num = len(enc) + 1
        return [float(i) for i in range(num)]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            num_data = self.num_train
            for num in range(num_data):
                enc = self.generate_input(0)
                dec = self.rule(enc)
                yield {
                    "inputs": enc,
                    "targets": dec,
                    "expected_attentions": self.generate_expected_attention(enc),
                    "expected_enc_attentions": self.generate_expected_enc_attention(enc)
                }
        elif dataset_split == problem.DatasetSplit.EVAL:
            num_data = self.num_eval
            for num in range(num_data):
                enc = self.generate_input(3)
                dec = self.rule(enc)
                yield {
                    "inputs": enc,
                    "targets": dec,
                    "expected_attentions": self.generate_expected_attention(enc),
                    "expected_enc_attentions": self.generate_expected_enc_attention(enc)
                }
        elif dataset_split == problem.DatasetSplit.TEST:
            num_data = self.num_test
            for num in range(num_data):
                for shard in range(6):
                    enc = self.generate_input(shard)
                    dec = self.rule(enc)
                    yield {
                        "inputs": enc,
                        "targets": dec,
                        "expected_attentions": self.generate_expected_attention(enc),
                        "expected_enc_attentions": self.generate_expected_enc_attention(enc)
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
            "shards": 6,
        }]

    def possible_len_sym_by_shard(self, shard):
        if shard == 0:
            return (1, 40), (1, 10)
        elif shard == 1:
            return (1, 40), (11, 20)
        elif shard == 2:
            return (1, 40), (1, 20)
        elif shard == 3:
            return (1, 400), (1, 10)
        elif shard == 4:
            return (1, 400), (11, 20)
        elif shard == 5:
            return (1, 400), (1, 20)

    def rule(self, enc_inp):
        return enc_inp

    def generate_input(self, shard):
        (min_len, max_len), (min_sym, max_sym) = self.possible_len_sym_by_shard(shard)
        symbols =string.ascii_lowercase[min_sym:max_sym+1]
        seqlen = np.random.randint(min_len, max_len + 1)
        enc = "".join([random.choice(symbols) for _ in range(seqlen)])
        return enc

@registry.register_problem
class AlgorithmicUnseenIdentityReverseAttention(AlgorithmicUnseenIdentity):
    def generate_expected_attention(self, enc):
        num = len(enc) + 1
        return [float(num-1-i) for i in range(num)]


@registry.register_problem
class AlgorithmicUnseenIdentityRandomAttention(AlgorithmicUnseenIdentity):
    def generate_expected_attention(self, enc):
        num = len(enc) + 1
        return [float(np.random.randint(0, num)) for _ in range(num)]



@registry.register_problem
class AlgorithmicSorting(AlgorithmicString):
    def example_reading_spec(self):
        data_fields = {
            "inputs": tf.VarLenFeature(tf.int64),
            "targets": tf.VarLenFeature(tf.int64),
            "expected_attentions": tf.VarLenFeature(tf.float32),
            "expected_enc_attentions": tf.VarLenFeature(tf.float32)
        }
        data_items_to_decoders = None
        return (data_fields, data_items_to_decoders)

    def generate_expected_enc_attention(self, enc):
        num = len(enc) + 1
        return [float(i) for i in range(num)]

    def generate_expected_attention(self, enc):
        return [float(i[0]) for i in sorted(enumerate(enc), key=lambda x:x[1])]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir

        if dataset_split == problem.DatasetSplit.TRAIN:
            num_data = self.num_train
            for num in range(num_data):
                enc = self.generate_input(0)
                dec = self.rule(enc)
                yield {
                    "inputs": enc,
                    "targets": dec,
                    "expected_attentions": self.generate_expected_attention(enc),
                    "expected_enc_attentions": self.generate_expected_enc_attention(enc)
                }
        elif dataset_split == problem.DatasetSplit.EVAL:
            num_data = self.num_eval
            for num in range(num_data):
                enc = self.generate_input(3)
                dec = self.rule(enc)
                yield {
                    "inputs": enc,
                    "targets": dec,
                    "expected_attentions": self.generate_expected_attention(enc),
                    "expected_enc_attentions": self.generate_expected_enc_attention(enc)
                }
        elif dataset_split == problem.DatasetSplit.TEST:
            num_data = self.num_test
            for num in range(num_data):
                for shard in range(6):
                    enc = self.generate_input(shard)
                    dec = self.rule(enc)
                    yield {
                        "inputs": enc,
                        "targets": dec,
                        "expected_attentions": self.generate_expected_attention(enc),
                        "expected_enc_attentions": self.generate_expected_enc_attention(enc)
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
            "shards": 6,
        }]

    @property
    def num_train(self):
        return 100000

    @property
    def num_eval(self):
        return 2000

    @property
    def num_test(self):
        return 2000

    def test_len_sym(self, shard):
        if shard == 0:
            return (1, 40), (1, 10)
        elif shard == 1:
            return (1, 40), (11, 20)
        elif shard == 2:
            return (1, 40), (1, 20)
        elif shard == 3:
            return (1, 400), (1, 10)
        elif shard == 4:
            return (1, 400), (11, 20)
        elif shard == 5:
            return (1, 400), (1, 20)

    def rule(self, enc_inp):
        return "".join(sorted(enc_inp))

    def generate_input(self, shard):
        (min_len, max_len), (min_sym, max_sym) = self.test_len_sym(shard)
        symbols =string.ascii_lowercase[min_sym:max_sym+1]
        seqlen = np.random.randint(min_len, max_len + 1)
        enc = "".join([random.choice(symbols) for _ in range(seqlen)])
        return enc

@registry.register_problem
class AlgorithmicSortingEnc(AlgorithmicSorting):
    def generate_expected_enc_attention(self, enc):
        return [float(i[0]) for i in sorted(enumerate(enc), key=lambda x:x[1])]

    def generate_expected_attention(self, enc):
        num = len(enc) + 1
        return [float(i) for i in range(num)]



@registry.register_problem
class AlgorithmicTranslation(AlgorithmicUnseenIdentity):
    def rule(self, enc_inp):
        matching = dict([(sym, idx) for idx, sym in enumerate(string.ascii_lowercase)])
        return "".join([string.ascii_lowercase[(matching[i]+10)%26] for i in enc_inp])


@registry.register_problem
class AlgorithmicUnseenReverse(AlgorithmicUnseenIdentity):
    def generate_expected_attention(self, enc):
        num = len(enc) + 1
        return [float(num-1-i) for i in range(num)]

    def generate_expected_enc_attention(self, enc):
        num = len(enc) + 1
        return [float(i) for i in range(num)]

    def rule(self, enc_inp):
        enc_len = len(enc_inp)
        return "".join([enc_inp[enc_len-1-i] for i in range(enc_len)])


@registry.register_problem
class AlgorithmicUnseenReverseEnc(AlgorithmicUnseenIdentity):
    def generate_expected_attention(self, enc):
        num = len(enc) + 1
        return [float(i) for i in range(num)]

    def generate_expected_enc_attention(self, enc):
        num = len(enc) + 1
        return [float(num-1-i) for i in range(num)]

    def rule(self, enc_inp):
        enc_len = len(enc_inp)
        return "".join([enc_inp[enc_len-1-i] for i in range(enc_len)])

@registry.register_problem
class AlgorithmicUnseenTiling(AlgorithmicString):
    @property
    def approx_vocab_size(self):
        return 2 ** 5  # ~8k

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
            "shards": 6,
        }]

    def possible_len_sym_by_shard(self, shard):
        if shard == 0:
            return (20, 40), (0, 10)
        elif shard == 1:
            return (20, 40), (10, 20)
        elif shard == 2:
            return (20, 40), (0, 20)
        elif shard == 3:
            return (20, 400), (0, 10)
        elif shard == 4:
            return (20, 400), (10, 20)
        elif shard == 5:
            return (20, 400), (0, 20)

    def generate_input(self, shard):
        (min_len, max_len), (min_sym, max_sym) = self.possible_len_sym_by_shard(shard)
        symbols = string.ascii_lowercase[min_sym:max_sym]
        seqlen = np.random.randint(min_len, max_len)
        pattern_len = np.random.randint(1, seqlen//3)
        pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
        res = ''
        while len(res) < 2*seqlen:
            res += pattern
        return res[:seqlen], res[seqlen:2*seqlen]

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        if dataset_split == problem.DatasetSplit.TRAIN:
            num_data = self.num_train
            for num in range(num_data):
                enc, dec = self.generate_input(0)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }
        elif dataset_split == problem.DatasetSplit.EVAL:
            num_data = self.num_eval
            for num in range(num_data):
                enc, dec = self.generate_input(0)
                yield {
                    "inputs": enc,
                    "targets": dec,
                }
        elif dataset_split == problem.DatasetSplit.TEST:
            num_data = self.num_test
            for num in range(num_data):
                for shard in range(6):
                    enc, dec = self.generate_input(shard)
                    yield {
                        "inputs": enc,
                        "targets": dec,
                    }


# class Tiling(text_problems.Text2TextProblem):
#     @property
#     def approx_vocab_size(self):
#         return 2 ** 5  # ~8k
#
#     @property
#     def is_generate_per_split(self):
#         return True
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 1,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 3,
#         }, {
#             "split": problem.DatasetSplit.TEST,
#             "shards": 3,
#         }]
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_pattern_lengths(self, dataset_split):
#         raise NotImplementedError()
#
#     def gen(self, pattern, total_len):
#         rep = (2*total_len) // len(pattern)
#         residue = 2*total_len-len(pattern)*rep
#         res = ''
#         while rep > 0:
#             rep-=1
#             res+=pattern
#         res = res + pattern[:residue]
#         return res[:total_len], res[total_len:]
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         lens = self.possible_pattern_lengths(dataset_split)
#         symbols = string.ascii_lowercase[:10]
#
#         if dataset_split == problem.DatasetSplit.TRAIN:
#             num_data = int(1e7)
#             for num in range(num_data):
#                 pattern_len = random.choice(lens)
#                 total_len = random.choice(range(pattern_len*2 + 1, 1+min(30, pattern_len*4)))
#                 pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
#                 enc, dec = self.gen(pattern, total_len)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }
#         else:
#             num_data = int(1e4)
#             for num in range(num_data):
#                 pattern_len = lens[num%3]
#                 total_len = random.choice(range(pattern_len*2 + 1, 1+min(30, pattern_len*4)))
#                 pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
#                 enc, dec = self.gen(pattern, total_len)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }

#     def eval_metrics(self):
#         return [metrics.Metrics.ACC, metrics.Metrics.ACC_PER_SEQ]

# @registry.register_problem
# class TilingExtension(Tiling):
#     def possible_pattern_lengths(self, dataset_split):
#         return [2,3,4,5,6,7,8,9] if dataset_split == problem.DatasetSplit.TRAIN else [10,11,12]
#
# @registry.register_problem
# class TilingInterpolation(Tiling):
#     def possible_pattern_lengths(self, dataset_split):
#         return [2,3,4,5,9,10,11,12] if dataset_split == problem.DatasetSplit.TRAIN else [6,7,8]
#
#
# @registry.register_problem
# class TilingSanity(text_problems.Text2TextProblem):
#     @property
#     def approx_vocab_size(self):
#         return 2 ** 5  # ~8k
#
#     @property
#     def is_generate_per_split(self):
#         return False
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 9,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 1,
#         }]
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_pattern_lengths(self, dataset_split):
#         raise NotImplementedError()
#
#     def gen(self, pattern, total_len):
#         rep = (2 * total_len) // len(pattern)
#         residue = 2 * total_len - len(pattern) * rep
#         res = ''
#         while rep > 0:
#             rep -= 1
#             res += pattern
#         res = res + pattern[:residue]
#         return res[:total_len], res[total_len:]
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         lens = [i for i in range(2, 13)]
#         symbols = string.ascii_lowercase[:10]
#
#         num_data = int(1e7)
#         for num in range(num_data):
#             pattern_len = random.choice(lens)
#             total_len = random.choice(range(pattern_len * 2 + 1, 1 + min(30, pattern_len * 4)))
#             pattern = "".join([random.choice(symbols) for _ in range(pattern_len)])
#             enc, dec = self.gen(pattern, total_len)
#             yield {
#                 "inputs": enc,
#                 "targets": dec,
#             }





# class SymbolBase(text_problems.Text2TextProblem):
#     @property
#     def is_generate_per_split(self):
#         # generate_data will shard the data into TRAIN and EVAL for us.
#         return True
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_lengths(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def possible_symbols(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def rule(self):
#         raise NotImplementedError()
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         min_len, max_len = self.possible_lengths(dataset_split)
#         min_sym, max_sym = self.possible_symbols(dataset_split)
#
#         if dataset_split == problem.DatasetSplit.TRAIN:
#             symbols = string.ascii_lowercase[:max_sym]
#             num_data = 100000
#             for num in range(num_data):
#                 seqlen = np.random.randint(min_len, max_len + 1)
#                 enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#                 dec = self.rule(enc)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }
#
#         else:
#             num_data = 10000
#             for num in range(num_data):
#                 for seqlen in range(min_len, max_len+1):
#                     for symlen in range(min_sym, max_sym+1):
#                         symbols = string.ascii_lowercase[:symlen]
#                         enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#                         dec = self.rule(enc)
#                         yield {
#                             "inputs": enc,
#                             "targets": dec,
#                         }

# class LengthBase(text_problems.Text2TextProblem):
#     @property
#     def approx_vocab_size(self):
#         return 2 ** 5  # ~8k
#
#     @property
#     def num_data(self, dataset_split):
#         raise NotImplementedError()
#
#     @property
#     def is_generate_per_split(self):
#         # generate_data will shard the data into TRAIN and EVAL for us.
#         return True
#
#     @property
#     def vocab_type(self):
#         return text_problems.VocabType.CHARACTER
#
#     def possible_lengths(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def possible_symbols(self, dataset_split):
#         #         if dataset_split == problem.DatasetSplit.TRAIN else 16
#         raise NotImplementedError()
#
#     def rule(self):
#         raise NotImplementedError()
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         raise NotImplementedError()
#
# class GeneralizeLengthTrain(LengthBase):
#     def num_data(self, dataset_split):
#         return int(1e6) if dataset_split == problem.DatasetSplit.TRAIN else int(1e4)
#     @property
#     def dataset_splits(self):
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 3,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 1,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12)
#     def possible_symbols(self, dataset_split):
#         return (10, 10)
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         min_len, max_len = self.possible_lengths(dataset_split)
#         min_sym, max_sym = self.possible_symbols(dataset_split)
#
#         num_data = self.num_data(dataset_split)
#         symbols = string.ascii_lowercase[:max_sym]
#         for num in range(num_data):
#             seqlen = np.random.randint(min_len, max_len + 1)
#             enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#             dec = self.rule(enc)
#             yield {
#                 "inputs": enc,
#                 "targets": dec,
#             }
#
# class GeneralizeLengthTest(LengthBase):
#     def num_data(self, dataset_split):
#         return 0 if dataset_split == problem.DatasetSplit.TRAIN else int(1e4)
#
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 1,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 6,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (13, 18)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10)
#
#     def generate_samples(self, data_dir, tmp_dir, dataset_split):
#         del data_dir
#         del tmp_dir
#
#         min_len, max_len = self.possible_lengths(dataset_split)
#         min_sym, max_sym = self.possible_symbols(dataset_split)
#
#         num_data = self.num_data(dataset_split)
#         symbols = string.ascii_lowercase[:max_sym]
#         seqlens = [i for i in range(min_len, max_len+1)]
#         # print(dataset_split)
#         # print("!!!!!!!!!!!!!!!!!!!!!!!!!")
#         # print(num_data)
#         for num in range(num_data):
#             for seqlen in seqlens:
#                 enc = "".join([random.choice(symbols) for _ in range(seqlen)])
#                 dec = self.rule(enc)
#                 yield {
#                     "inputs": enc,
#                     "targets": dec,
#                 }


# @registry.register_problem
# class GeneralizeSymbolsCyclic(SymbolBase):
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 10,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 5,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (5, 12)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10) if dataset_split == problem.DatasetSplit.TRAIN else (11, 15)
#
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
#
# @registry.register_problem
# class GeneralizeSymbolsReverse(SymbolBase):
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 10,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 5,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (5, 12)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10) if dataset_split == problem.DatasetSplit.TRAIN else (11, 15)
#
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
#
# @registry.register_problem
# class GeneralizeSymbolsIdentity(SymbolBase):
#     @property
#     def dataset_splits(self):
#         """Splits of data to produce and number of output shards for each."""
#         # 10% evaluation data
#         return [{
#             "split": problem.DatasetSplit.TRAIN,
#             "shards": 10,
#         }, {
#             "split": problem.DatasetSplit.EVAL,
#             "shards": 5,
#         }]
#
#     def possible_lengths(self, dataset_split):
#         return (1, 12) if dataset_split == problem.DatasetSplit.TRAIN else (5, 12)
#
#     def possible_symbols(self, dataset_split):
#         return (10, 10) if dataset_split == problem.DatasetSplit.TRAIN else (11, 15)
#
#     def rule(self, enc_inp):
#         return enc_inp


# @registry.register_problem
# class GeneralizeLengthCyclicSanity(GeneralizeLengthTrain):
#     def possible_lengths(self, dataset_split):
#         return (1, 18)
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
# @registry.register_problem
# class GeneralizeLengthCyclicTrain(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
# @registry.register_problem
# class GeneralizeLengthCyclicTest(GeneralizeLengthTest):
#     def rule(self, enc_inp):
#         return enc_inp[-1] + enc_inp[:-1]
#
#
# @registry.register_problem
# class GeneralizeLengthReverseSanity(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
#     def possible_lengths(self, dataset_split):
#         return (1, 18)
#
# @registry.register_problem
# class GeneralizeLengthReverseTrain(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
# @registry.register_problem
# class GeneralizeLengthReverseTest(GeneralizeLengthTest):
#     def rule(self, enc_inp):
#         return "".join([enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))])
#
# @registry.register_problem
# class GeneralizeLengthIdentitySanity(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return enc_inp
#
#     def possible_lengths(self, dataset_split):
#         return (1, 18)
#
# @registry.register_problem
# class GeneralizeLengthIdentityTrain(GeneralizeLengthTrain):
#     def rule(self, enc_inp):
#         return enc_inp
#
# @registry.register_problem
# class GeneralizeLengthIdentityTest(GeneralizeLengthTest):
#     def rule(self, enc_inp):
#         return enc_inp
