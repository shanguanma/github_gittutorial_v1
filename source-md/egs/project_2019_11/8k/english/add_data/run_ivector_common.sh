#!/bin/bash

# Copyright 2017  Hossein Hadian
#           2017  Vimal Manohar
# Apache 2.0
# md note
# note: this script has not generate_aligments stage.
#. ./cmd.sh
#set -e
stage=0
speed_perturb=true
train_set=seame_ubs_train_cs200_cs_15_i2r_mandarin_hkust_dt_man200_imda_i2r_msf_imda_boundarymic_codec  # Supervised training set
ivector_train_set=  # data set for training i-vector extractor. 
                    # If not provided, train_set will be used.
cmd="slurm.pl --quiet --exclude=node05,node06"
nnet3_affix=_ubs2020_cs_big
exp_root=feat_log

. ./path.sh
. ./utils/parse_options.sh

# combine mandary data
mandarin_data=hkust_i2r-mandarin_dt_man200_cs_15_8k_from_ASRUman500

if [ $stage -le 0 ];then
   utils/fix_data_dir.sh data/cs_15_8k_from_ASRUman500
   utils/fix_data_dir.sh data/dt_man200_8k
   utils/fix_data_dir.sh data/hkust_8k 
   utils/fix_data_dir.sh data/i2r-mandarin_8k 
   utils/combine_data.sh  data/$mandarin_data \
            data/cs_15_8k_from_ASRUman500 \
            data/dt_man200_8k \
            data/hkust_8k \
            data/i2r-mandarin_8k 
  utils/fix_data_dir.sh data/$mandarin_data 
  utils/validate_data_dir.sh  data/$mandarin_data 

fi


# perturbed data preparation
if [ "$speed_perturb" == "true" ]; then
  if [ $stage -le 1 ]; then
    # Although the nnet will be trained by high resolution data, we still have
    # to perturb the normal data to get the alignments.
    # _sp stands for speed-perturbed

    for datadir in ${mandarin_data} ; do
      utils/data/perturb_data_dir_speed_3way.sh  data/${datadir} data/${datadir}_sp  
      utils/fix_data_dir.sh data/${datadir}_sp

      mfccdir=feat/mfcc_perturbed
      steps/make_mfcc.sh --cmd "$cmd" --nj 50 \
        data/${datadir}_sp $exp_root/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh \
        data/${datadir}_sp $exp_root/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_sp
    done
  fi
  mandarin_data=${mandarin_data}_sp
  
fi

# it is used to ali lattice 
# it is 13 mfcc
train_sp_dir=data/${train_set}_sp
if [ $stage -le 2 ];then
    utils/combine_data.sh $train_sp_dir \
                 data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec \
                 data/hkust_i2r-mandarin_dt_man200_cs_15_8k_from_ASRUman500_sp \
                 data/seame_trainset_8k_cs200_8k_sp_ubs2020_train 
    utils/fix_data_dir.sh $train_sp_dir
    utils/validate_data_dir.sh  $train_sp_dir
 
fi
if [ $stage -le 3 ]; then
  mfccdir=feat/mfcc_hires
  
  for dataset in  $mandarin_data; do
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$cmd" data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
fi
#  for dataset in dev_sge dev_man; do
    # Create MFCCs for the eval set
#    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
#    steps/make_mfcc.sh --cmd "$cmd" --nj 10 --mfcc-config conf/mfcc_hires.conf \
#        data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
#    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
#    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
#  done
#fi

# combine final train data
# it is 40 dim mfcc
# it is used to train chain model
english_data_sp_hires_dir=data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires
cs_data_hires_dir=data/seame_trainset_8k_cs200_8k_sp_hires_ubs2020_train_hires
if [ $stage -le 4 ];then
   utils/combine_data.sh data/${train_set}_sp_hires \
                   $english_data_sp_hires_dir \
                   $cs_data_hires_dir \
                   data/hkust_i2r-mandarin_dt_man200_cs_15_8k_from_ASRUman500_sp_hires
   utils/fix_data_dir.sh data/${train_set}_sp_hires
   utils/validate_data_dir.sh  data/${train_set}_sp_hires
fi 




if [ -z "$ivector_train_set" ]; then
  ivector_train_set=${train_set}_sp
fi

# ivector extractor training
if [ $stage -le 5 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    data/${ivector_train_set}_hires \
    feat/nnet3${nnet3_affix}/pca_transform
fi

if [ $stage -le 6 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj 50 --num-frames 200000 \
    data/${ivector_train_set}_hires 512 \
    feat/nnet3${nnet3_affix}/pca_transform feat/nnet3${nnet3_affix}/diag_ubm
fi

if [ $stage -le 7 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj 50 \
    data/${ivector_train_set}_hires feat/nnet3${nnet3_affix}/diag_ubm \
    feat/nnet3${nnet3_affix}/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the ${train_set} data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    data/${ivector_train_set}_hires data/${ivector_train_set}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 50 \
    data/${ivector_train_set}_max2_hires feat/nnet3${nnet3_affix}/extractor \
    data/nnet3${nnet3_affix}/ivectors_${ivector_train_set}_hires || exit 1;
fi

test_set_1="wiz3ktest dev_imda_part3_ivr"
test_set_2="msf_baby_bonus-8k  ubs2020_dev_cs"
test_set_3="ubs2020_dev_eng ubs2020_dev_man"

if [ $stage -le 9 ]; then
  for dataset in $test_set_1 $test_set_2 $test_set_3; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      data/${dataset}_hires feat/nnet3${nnet3_affix}/extractor \
      data/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done
fi

exit 0;
