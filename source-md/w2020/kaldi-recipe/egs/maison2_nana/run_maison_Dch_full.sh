#!/bin/bash

# original kaldi path:/home/nana511/maison2/kaldi-data/
#                     model-ch-80h is the selected 80 hours data
#                     model-ch-all is the whole data
#                     I have splited every channel into a seperate sub-folder 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=120 
chain_stage=0
chain_model_train_stage=-10
decode_nj=96
nj=20

# releated  ngram 
n_order=4          # order of  n-gram lm, for example 4-gram ,its order is 4, default is 4,  you usually can set 3 or 4.
oov_symbol="<UNK>" #  oov symbol for making  maxent lm.

# releated dict
use_pp=true             # we compute the pronunciation and silence probabilities from training data,
                        # and re-create the lang and lang_test directory.
# realted gmm 
shortest_utt_num=20000  # these utterances is used to do train mono, it is useful for alignment. this value
                   # it is usually about 1/5 the entire train set text utterance.

wave_sample=16k     # wave sample frequency, it may be 8k or 16k
# releated chain model
feats_type=perturb  # using speed perturb and volumn perturb for data augmentation
                    # default is perturb
nnet3_affix="1a"      # i-vector folder affix
sp_suffix=_sp       # # when using speed perturb and volumn perturb. it should be "_sp" ,else "_nosp" 
small_dnn=true     # if train data less 50  hours ,it will be true, else , it will false.
# [Task dependent] Set the datadir name created by
src_train_data_dir=data/model-ch-all
src_test_data_dir=data/dev
train_set=data_Dch                  # train set name
test_sets="data_Ach data_Bch  data_Cch  data_Dch  data_Ech  data_Fch  data_Gch  data_Hch  data_src"
srcdictdir=test_1/maison2_dict
lm_train_text=              # Use  train set text to do  lm training if not specified.
tgtdir=train_data_Dch_new       #  root folder of the entire asr system 
suffix="_Dch"               # you can set it, then run different model in same tgtdir,
                            # here I will set it as the channel flag, for Ach, Bch, etc
. utils/parse_options.sh

./source-md/w2020/kaldi-recipe/egs/maison2_nana/kaldi_asr.sh               \
                                                  --stage $stage           \
                                                  --stop_stage $stop_stage \
                                                  --chain_stage $chain_stage \
                                                  --chain_model_train_stage $chain_model_train_stage \
                                                  --decode_nj $decode_nj \
                                                  --nj $nj                 \
                                                  --n_order $n_order       \
                                                  --use_pp $use_pp         \
                                                  --shortest_utt_num $shortest_utt_num \
                                                  --wave_sample $wave_sample \
                                                  --feats_type $feats_type  \
                                                  --nnet3_affix $nnet3_affix \
                                                  --sp_suffix $sp_suffix \
                                                  --src_train_data_dir "$src_train_data_dir" \
                                                  --src_test_data_dir "$src_test_data_dir" \
                                                  --train_set "$train_set" \
                                                  --test_sets "$test_sets" \
                                                  --srcdictdir "$srcdictdir" \
                                                  --tgtdir "$tgtdir" \
                                                  --suffix $suffix 
