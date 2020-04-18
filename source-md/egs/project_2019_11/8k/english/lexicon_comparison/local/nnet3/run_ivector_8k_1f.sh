#!/bin/bash

# Copyright 2017  Hossein Hadian
#           2017  Vimal Manohar
# Apache 2.0
# note:
# 1.ivector train set is train or subset of train.
# 2.using pca transform matrix
# 3. we still haven't to perturb the normal data to get the alignments. 

# 4. support three test set
# 5. Not using $ tgtdir
# 6. ivector-extractor , pca, ubm and etc are stored at feat/nnet3
# 7. ivector-feature is stored as data/nnet3
# 8. 40dim _sp_hires and 13 dim _sp are stored as data/
# 9 all related log is stored as feat_log/
. ./cmd.sh
set -e
stage=1
cmd="slurm.pl --quiet --exclude=node06,node07"
train_set=train_imda_part3_ivr_msf_i2r   # Supervised training set
test_set_1=
test_set_2=
test_set_3=
test_set_is_run_mfcc_hires=false  # if test_set mfcc-hires have exist, you can set false, it don't need to run it again.
codec_data=imda_boundarymic_codec
lores_train_data_dir=data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec
train_data_sp_hires_dir=data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires
nnet3_affix=_1d
. ./path.sh
. ./utils/parse_options.sh
exp_root=feat_log

 
# combine $codec_data and ${datadir}_sp  as lores_train_data to do align lat. 
if [ $stage -le 1 ];then
   utils/fix_data_dir.sh  data/$codec_data
   utils/fix_data_dir.sh data/${train_set}_sp 
   utils/validate_data_dir.sh  data/$codec_data
   utils/validate_data_dir.sh data/${train_set}_sp
   utils/combine_data.sh $lores_train_data_dir  data/${train_set}_sp  data/$codec_data
   # because combine_data.sh has done fix_data_dir.sh
   utils/validate_data_dir.sh $lores_train_data_dir
fi
# get codec_data 40 dim feature
if [ $stage -le 2 ]; then
  mfccdir=feat/mfcc_hires
  for dataset in $codec_data; do
    utils/fix_data_dir.sh data/$dataset 
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    #utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$cmd" data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
fi


if [ $stage -le 3 ];then
   
   utils/combine_data.sh  $train_data_sp_hires_dir  data/${train_set}_sp_hires data/${codec_data}_hires
   utils/validate_data_dir.sh $train_data_sp_hires_dir 
fi


# ivector extractor training
if [ $stage -le 4 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    $train_data_sp_hires_dir \
    feat/nnet3${nnet3_affix}/pca_transform
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj 30 --num-frames 200000 \
    $train_data_sp_hires_dir 512 \
    feat/nnet3${nnet3_affix}/pca_transform feat/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj 10 \
    $train_data_sp_hires_dir feat/nnet3${nnet3_affix}/diag_ubm \
    feat/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # We extract iVectors on all the ${train_set} data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $train_data_sp_hires_dir ${train_data_sp_hires_dir}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 30 \
    ${train_data_sp_hires_dir}_max2_hires feat/nnet3${nnet3_affix}/extractor \
    data/nnet3${nnet3_affix}/ivectors_train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires  || exit 1;
fi


if [ "$test_set_is_run_mfcc_hires" == "true" ]; then
 if [ $stage -le 8 ];then
  mfccdir=feat/mfcc_hires
  for dataset in $test_set_1 $test_set_2; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$cmd" --nj 10 --mfcc-config conf/mfcc_hires_8k.conf \
        data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
 fi
fi
if [ $stage -le 9 ]; then
  for dataset in $test_set_1 $test_set_2; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      data/${dataset}_hires feat/nnet3${nnet3_affix}/extractor \
      data/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done
fi


if [ $stage -le 10 ];then
  mfccdir=feat/mfcc_hires
  for dataset in $test_set_3; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$cmd" --nj 10 --mfcc-config conf/mfcc_hires_8k.conf \
        data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
fi
# get test_set_3 ivector feature
if [ $stage -le 11 ]; then
  for dataset in $test_set_3; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      data/${dataset}_hires feat/nnet3${nnet3_affix}/extractor \
      data/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done
fi

