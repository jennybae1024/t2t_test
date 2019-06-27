import re

from gutenberg import acquire
from gutenberg import cleanup

from tensor2tensor.data_generators import problem
from tensor2tensor.data_generators import text_problems
from tensor2tensor.utils import registry
<<<<<<< HEAD
#test
=======

>>>>>>> cd74e78a5381e1f5251dae985b0ad7e93c5393e6

@registry.register_problem
class NumberSeqs(text_problems.Text2TextProblem):
    
    @property
    def approx_vocab_size(self):
        return 2 ** 4  
    @property
    def vocab_type(self):
        return 'character'
    
    @property
    def is_generate_per_split(self):
        return True



    def generate_samples(self, data_dir, tmp_dir, dataset_split):
        del data_dir
        del tmp_dir
        ftn = lambda enc_inp: [enc_inp[len(enc_inp) - 1 - i] for i in range(len(enc_inp))]
        max_len = 12 if dataset_split == problem.DatasetSplit.TRAIN else 16
        num_data = 10e6 if dataset_split == problem.DatasetSplit.TRAIN else 10e4
        for _ in range(num_data):
            seqlen = np.random.randint(1, max_len + 1)
            enc_list = [str(i) for i in list(np.random.randint(2, 10, seqlen))]
            dec_list = ftn(enc_list)
            yield {"inputs": "".join(enc_list), "targets": "".join(dec_list)}
    
                   
                   
                   