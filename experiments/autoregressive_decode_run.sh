python autoregressive_decode.py --problem=algorithmic_unseen_identity --model=transformer --hparams_set=transformer_sym_noise_tiny --test_shard=0 --global_steps=16000 --gpu_num=1
python autoregressive_decode.py --problem=algorithmic_unseen_identity --model=transformer --hparams_set=transformer_sym_noise_tiny --test_shard=0 --global_steps=32000 --gpu_num=1
python basic_train.py --problem=algorithmic_unseen_identity --model_name=lstm_seq2seq_attention_bidirectional_encoder --hparam_set=lstm_bahdanau_attention --train_steps=32000 --fresh=1 --gpu_num=1 > auto_run1.log
python basic_train.py --problem=algorithmic_sorting --model_name=lstm_seq2seq_attention_bidirectional_encoder --hparam_set=lstm_bahdanau_attention --train_steps=32000 --fresh=1 --gpu_num=2 > auto_run2.log
