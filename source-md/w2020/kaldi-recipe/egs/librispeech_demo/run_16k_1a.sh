#!/bin/bash

# original kaldi path; /home/hhx502/w2020/projects/malay_cts/data 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=120 
chain_stage=0
chain_model_train_stage=-10
num_epochs=20
decode_nj=40
nj=20

# releated  ngram 
n_order=4          # order of  n-gram lm, for example 4-gram ,its order is 4, default is 4,  you usually can set 3 or 4.
oov_symbol="<UNK>" #  oov symbol for making  maxent lm.

# releated dict
use_pp=true             # we compute the pronunciation and silence probabilities from training data,
                        # and re-create the lang and lang_test directory.
# realted gmm 
shortest_utt_num=10000  # these utterances is used to do train mono, it is useful for alignment. this value
                   # it is usually about 1/5 the entire train set text utterance.

wave_sample=16k     # wave sample frequency, it may be 8k or 16k
# releated chain model
feats_type=perturb  # using speed perturb and volumn perturb for data augmentation
                    # default is perturb
nnet3_affix="1a"      # i-vector folder affix
sp_suffix=_sp       # # when using speed perturb and volumn perturb. it should be "_sp" ,else "_nosp" 
#dnn_size=normal       # if train data less 100  hours ,it will be small, else , it will normal.
dnn_size=small 

# [Task dependent] Set the datadir name created by
src_train_data_dir=kaldi_data
src_test_data_dir=kaldi_data
train_set=train_clean_100                 # train set name
#test_sets="test_8khp"
test_sets="dev_clean  dev_other  test_clean  test_other"
srcdictdir=kaldi_data/dict_nosp
lm_train_text=              # Use  train set text to do  lm training if not specified.
tgtdir=run_16k_1a       #  root folder of the entire asr system 
suffix="_1a"               # you can set it, then run different model in same tgtdir,
                            # here I will set it as the channel flag, for Ach, Bch, etc
. utils/parse_options.sh



./source-md/w2020/kaldi-recipe/egs/librispeech_demo/kaldi_asr_v1.sh               \
                                                  --stage $stage           \
                                                  --stop_stage $stop_stage \
                                                  --chain_stage $chain_stage \
                                                  --chain_model_train_stage $chain_model_train_stage \
                                                  --num_epochs ${num_epochs} \
                                                  --decode_nj $decode_nj \
                                                  --nj $nj                 \
                                                  --n_order $n_order       \
                                                  --use_pp $use_pp         \
                                                  --shortest_utt_num $shortest_utt_num \
                                                  --wave_sample $wave_sample \
                                                  --feats_type $feats_type  \
                                                  --nnet3_affix $nnet3_affix \
                                                  --sp_suffix $sp_suffix \
                                                  --dnn_size $dnn_size   \
                                                  --src_train_data_dir "$src_train_data_dir" \
                                                  --src_test_data_dir "$src_test_data_dir" \
                                                  --train_set "$train_set" \
                                                  --test_sets "$test_sets" \
                                                  --srcdictdir "$srcdictdir" \
                                                  --tgtdir "$tgtdir" \
                                                  --suffix $suffix 

