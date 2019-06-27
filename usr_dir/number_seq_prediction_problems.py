import re
from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
import numpy as np
import random
import string    

@registry.register_problem
class ReverseSeq(text_problems.Text2TextProblem):    
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
        }]    
    
    @property
    def approx_vocab_size(self):
        return 2 ** 4  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return True

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        ftn = lambda enc_inp: [enc_inp[len(enc_inp)-1-i] for i in range(len(enc_inp))]
        max_len = 12 if dataset_split == problem.DatasetSplit.TRAIN else 16
        num_data = int(1e7) if dataset_split == problem.DatasetSplit.TRAIN else int(1e5)
        for num in range(num_data):
            seqlen = np.random.randint(1, max_len+1)
            enc_list = [str(i) for i in list(np.random.randint(0, 10, seqlen))]
            dec_list = ftn(enc_list)
            yield {
                  "inputs": "".join(enc_list),
                  "targets": "".join(dec_list),
              }


            
            
    
@registry.register_problem
class CopySeq(text_problems.Text2TextProblem):    
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
        }]    
    
    @property
    def approx_vocab_size(self):
        return 2 ** 4  # ~8k

    @property
    def is_generate_per_split(self):
        # generate_data will shard the data into TRAIN and EVAL for us.
        return True

    @property
    def vocab_type(self):
        return text_problems.VocabType.CHARACTER

    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        ftn = lambda enc_inp: enc_inp
        max_len = 12 if dataset_split == problem.DatasetSplit.TRAIN else 16
        num_data = int(1e7) if dataset_split == problem.DatasetSplit.TRAIN else int(1e5)
        for num in range(num_data):
            seqlen = np.random.randint(1, max_len+1)
            enc_list = [str(i) for i in list(np.random.randint(0, 10, seqlen))]
            dec_list = ftn(enc_list)
            yield {
                  "inputs": "".join(enc_list),
                  "targets": "".join(dec_list),
              }    
                   
                   