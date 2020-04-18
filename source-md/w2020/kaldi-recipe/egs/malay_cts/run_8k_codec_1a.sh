#!/bin/bash

# original kaldi path; /home/hhx502/w2020/projects/malay_cts/data 

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail


stage=1
stop_stage=2
asr_stage=2
asr_stop_stage=120 
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

wave_sample=8k     # wave sample frequency, it may be 8k or 16k
# releated chain model
feats_type=perturb  # using speed perturb and volumn perturb for data augmentation
                    # default is perturb
frontend_type=codec
nnet3_affix="1a"      # i-vector folder affix
sp_suffix=_sp       # # when using speed perturb and volumn perturb. it should be "_sp" ,else "_nosp" 
#dnn_size=normal       # if train data less 100  hours ,it will be small, else , it will normal.
dnn_size=small 

# [Task dependent] Set the datadir name created by
src_train_data_dir=kaldi_data/8k
src_test_data_dir=kaldi_data/8k
train_set=train8kmic_codec        # train set name
#test_sets="test_8khp"
test_sets="test8kmic"
srcdictdir=kaldi_data/16k/dict_16k
lm_train_text=              # Use  train set text to do  lm training if not specified.
tgtdir=run_8k_codec_1a       #  root folder of the entire asr system 
suffix="_1a"               # you can set it, then run different model in same tgtdir,
                            # here I will set it as the channel flag, for Ach, Bch, etc

# related codec
codec_stage=1
codec_stop_stage=120
codec_list=/home4/md510/package/source-md/asr_frontend/data_level/codec/codec-list-full.txt
sampling_rate=8000
src_data_dir=kaldi_data/8k/train8kmic
tgt_data_dir=run_8k_codec_1a/data_1a/train8kmic_codec  # set it  by yourself 



. utils/parse_options.sh

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
   ./source-md/asr_frontend/data_level/codec/add-codec_v1.sh \
         --stage ${codec_stage} \
         --stop_stage ${codec_stop_stage} \
         --codec_list ${codec_list} \
         --sampling_rate ${sampling_rate} \
         --src_data_dir "${src_data_dir}" \
         --tgt_data_dir "${tgt_data_dir}"
fi
if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  ./source-md/w2020/kaldi-recipe/egs/malay_cts/kaldi_asr_v1.sh               \
                                                  --stage $asr_stage           \
                                                  --stop_stage $asr_stop_stage \
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
                                                  --frontend_type ${frontend_type} \
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
fi
