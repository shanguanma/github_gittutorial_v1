#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

stage=1
stop_stage=12
ngpu=1
nj=16
dumpdir=dump_1a     # Directory to dump features.
expdir=exp_1a      # Directory to save experiments.

train_set=train_trn
dev_set=train_dev
#dev_set=
eval_sets="dev_man dev_sge"
# path_v3.sh
# run nodeo5 gpu=2
#asr_config=conf/espnet2_new_conf/training/train_asr_transformer_aishell_batchsize64_xavier_uniform_warmup_speaug.yaml       
asr_config=conf/espnet2_new_conf/training/train_asr_transformer_aishell_batchsize64_xavier_uniform_reducelronplateau_speaug.yaml 
#asr_config=conf/espnet2_conf/training/train_asr_transformer_aishell_batchsize64_xavier_uniform_warmup_ag4_epoch50.yaml
decode_config=conf/espnet2_new_conf/decode/decode_pytorch_transformer_csj_v2.yaml 

lm_config=conf/espnet2_new_conf/lm/train_seq_rnn_lm.yaml 
use_lm=true
use_wordlm=false
gpu_decode=true
#asr_speech_fold_length=800 # fold_length for speech data during ASR training
#asr_text_fold_length=150   # fold_length for text data during ASR training
#lm_fold_length=150         # fold_length for LM training
# speed perturbation related
# (train_set will be "${train_set}_sp" if speed_perturb_factors is specified)
speed_perturb_factors="0.9 1.0 1.1"

. utils/parse_options.sh

source-md/w2020/espnet-recipe/seame_e2e_epsnet2_v1/asr.sh \
    --stage ${stage}                            \
    --stop_stage ${stop_stage}                  \
    --ngpu ${ngpu}                              \
    --nj ${nj}                                  \
    --dumpdir "${dumpdir}"                      \
    --expdir "${expdir}"                        \
    --audio_format wav                          \
    --feats_type fbank_pitch                    \
    --token_type bpe                            \
    --nbpe 3000                                 \
    --bpemode bpe                               \
    --use_lm ${use_lm}                          \
    --use_word_lm ${use_wordlm}                 \
    --lm_config "${lm_config}"                  \
    --asr_config "${asr_config}"                \
    --decode_config "${decode_config}"          \
    --train_set "${train_set}"                  \
    --dev_set "${dev_set}"                      \
    --eval_sets "${eval_sets}"                  \
    --srctexts "data/${train_set}/text" "$@"    \
    --speed_perturb_factors "${speed_perturb_factors}"
