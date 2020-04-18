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
speed_perturb=true
train_set=train  # Supervised training set
test_set_1=
test_set_2=
test_set_3=
ivector_train_set=  # data set for training i-vector extractor. 
                    # If not provided, train_set will be used.
nnet3_affix=
. ./path.sh
. ./utils/parse_options.sh
exp_root=feat_log
# perturbed data preparation
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    # Although the nnet will be trained by high resolution data
    # _sp stands for speed-perturbed

    for datadir in ${train_set} ${ivector_train_set}; do
      utils/data/perturb_data_dir_speed_3way.sh data/${datadir} data/${datadir}_sp
      utils/fix_data_dir.sh data/${datadir}_sp
 
      mfccdir=feat/mfcc_perturbed
      steps/make_mfcc.sh --cmd "$cmd" --nj 40 --mfcc-config conf/mfcc_8k.conf \
        data/${datadir}_sp $exp_root/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh \
        data/${datadir}_sp $exp_root/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_sp
    done
  fi
  train_set=${train_set}_sp
  if ! [ -z "$ivector_train_set" ]; then
    ivector_train_set=${ivector_train_set}_sp
  fi
fi
if [ "$speed_perturb" == "false" ];then
   affix=_nosp
else
   affix=""
fi

if [ $stage -le 2 ]; then
  mfccdir=feat/mfcc_hires$affix
  for dataset in $ivector_train_set $train_set; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires$affix
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires$affix

    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$cmd" data/${dataset}_hires$affix $exp_root/make_hires$affix/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires$affix $exp_root/make_hires$affix/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires$affix;
  done

  for dataset in $test_set_1 $test_set_2 $test_set_3; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires$affix
    steps/make_mfcc.sh --cmd "$cmd" --nj 10 --mfcc-config conf/mfcc_hires_8k.conf \
        data/${dataset}_hires$affix $exp_root/make_hires$affix/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires$affix $exp_root/make_hires$affix/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires$affix  # remove segments with problems
  done
fi
if [ -z "$ivector_train_set" ]; then
  ivector_train_set=$train_set
fi

# ivector extractor training
if [ $stage -le 4 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    data/${ivector_train_set}_hires$affix \
    feat/nnet3${nnet3_affix}/pca_transform
fi

if [ $stage -le 5 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj 30 --num-frames 200000 \
    data/${ivector_train_set}_hires$affix 512 \
    feat/nnet3${nnet3_affix}/pca_transform feat/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 6 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj 10 \
    data/${ivector_train_set}_hires$affix feat/nnet3${nnet3_affix}/diag_ubm \
    feat/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 7 ]; then
  # We extract iVectors on all the ${train_set} data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${ivector_train_set}_hires$affix data/${ivector_train_set}_max2_hires$affix

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 30 \
    data/${ivector_train_set}_max2_hires$affix feat/nnet3${nnet3_affix}/extractor \
    data/nnet3${nnet3_affix}/ivectors_${ivector_train_set}_hires$affix || exit 1;
fi

if [ $stage -le 8 ]; then
  for dataset in $test_set_1 $test_set_2 $test_set_3; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      data/${dataset}_hires$affix feat/nnet3${nnet3_affix}/extractor \
      data/nnet3${nnet3_affix}/ivectors_${dataset}_hires$affix || exit 1;
  done
fi

exit 0;
