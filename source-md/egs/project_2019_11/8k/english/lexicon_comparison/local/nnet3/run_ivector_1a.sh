#!/bin/bash

# refrence:https://github.com/kaldi-asr/kaldi/blob/master/egs/swbd/s5c/local/nnet3/run_ivector_common.sh
. ./cmd.sh
set -e
stage=1
train_stage=-10
generate_alignments=true
speed_perturb=true
tgtdir=
train_set=
test_set_1=
test_set_2=
lang=
. ./path.sh
. ./utils/parse_options.sh

mkdir -p  $tgtdir/exp/nnet3

# train lda_mllt , it is sub set from full train set.
train_lda_mllt_data=${train_set}_200k_nodup 
if $speed_perturb; then
  if [ $stage -le 1 ]; then
    # Although the nnet will be trained by high resolution data, we still have
    # to perturb the normal data to get the alignments _sp stands for
    # speed-perturbed
    echo "$0: preparing directory for speed-perturbed data"
    utils/data/perturb_data_dir_speed_3way.sh --always-include-prefix true \
           $tgtdir/data/${train_set}  $tgtdir/data/${train_set}_sp

    echo "$0: creating MFCC features for low-resolution speed-perturbed data"
    mfccdir=$tgtdir/mfcc_perturbed
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 50 --mfcc-config conf/mfcc_8k.conf \
                        $tgtdir/data/${train_set}_sp  $tgtdir/exp/make_mfcc/${train_set}_sp $mfccdir
    steps/compute_cmvn_stats.sh  $tgtdir/data/${train_set}_sp  $tgtdir/exp/make_mfcc/${train_set}_sp $mfccdir
    utils/fix_data_dir.sh  $tgtdir/data/${train_set}_sp
  fi

  if [ $stage -le 2 ] && $generate_alignments; then
    # obtain the alignment of the perturbed data
    steps/align_fmllr.sh --nj 100 --cmd "$train_cmd" \
       $tgtdir/data/${train_set}_sp  $tgtdir/data/lang  $tgtdir/exp/tri4  $tgtdir/exp/tri4_ali_nodup_sp
  fi
fi

train_set_sp=${train_set}_sp

if [ $stage -le 3 ]; then
  mfccdir=$tgtdir/mfcc_hires
  # the 100k_nodup directory is copied seperately, as
  # we want to use exp/tri1_ali for lda_mllt training
  # the main train directory might be speed_perturbed
  for dataset in $train_set_sp $train_lda_mllt_data; do
    utils/copy_data_dir.sh $tgtdir/data/$dataset $tgtdir/data/${dataset}_hires

    utils/data/perturb_data_dir_volume.sh $tgtdir/data/${dataset}_hires

    steps/make_mfcc.sh --nj 70 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$train_cmd" $tgtdir/data/${dataset}_hires $tgtdir/exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh $tgtdir/data/${dataset}_hires $tgtdir/exp/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh $tgtdir/data/${dataset}_hires;
  done
fi
if [ $stage -le 4 ];then  
  for dataset in $test_set_1 $test_set_2; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh $tgtdir/data/$dataset $tgtdir/data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 10 --mfcc-config conf/mfcc_hires_8k.conf \
        $tgtdir/data/${dataset}_hires $tgtdir/exp/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh $tgtdir/data/${dataset}_hires $tgtdir/exp/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh $tgtdir/data/${dataset}_hires  # remove segments with problems
  done

  # Take the first 30k utterances (about 1/8th of the data) this will be used
  # for the diagubm training
  utils/subset_data_dir.sh --first $tgtdir/data/${train_set_sp}_hires 30000 $tgtdir/data/${train_set_sp}_30k_hires
  utils/data/remove_dup_utts.sh 200 $tgtdir/data/${train_set_sp}_30k_hires $tgtdir/data/${train_set_sp}_30k_nodup_hires  
fi


# ivector extractor training
if [ $stage -le 5 ]; then
  # We need to build a small system just because we need the LDA+MLLT transform
  # to train the diag-UBM on top of.  We use --num-iters 13 because after we get
  # the transform (12th iter is the last), any further training is pointless.
  # this decision is based on fisher_english
  steps/train_lda_mllt.sh --cmd "$train_cmd" --num-iters 13 \
    --splice-opts "--left-context=3 --right-context=3" \
    5500 90000 $tgtdir/data/$train_lda_mllt_data \
    $lang $tgtdir/exp/tri1_ali_200k_nodup  $tgtdir/exp/nnet3/tri2b
fi

if [ $stage -le 6 ]; then
  # To train a diagonal UBM we don't need very much data, so use the smallest subset.
  nj=$(wc -l $tgtdir/data/${train_set_sp}_30k_nodup_hires/spk2utt | awk '{print $1}' || exit 1;)
  steps/online/nnet2/train_diag_ubm.sh --cmd "$train_cmd" --nj $nj --num-frames 20000 \
    $tgtdir/data/${train_set_sp}_30k_nodup_hires 512 $tgtdir/exp/nnet3/tri2b $tgtdir/exp/nnet3/diag_ubm
fi

if [ $stage -le 7 ]; then
  # iVector extractors can be sensitive to the amount of data, but this one has a
  # fairly small dim (defaults to 100) so we don't use all of it, we use just the
  # 100k subset (just under half the data).
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$train_cmd" --nj 10 \
    $tgtdir/data/${train_lda_mllt_data}_hires $tgtdir/exp/nnet3/diag_ubm $tgtdir/exp/nnet3/extractor || exit 1;
fi

if [ $stage -le 8 ]; then
  # We extract iVectors on all the train_nodup data, which will be what we
  # train the system on.

  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 $tgtdir/data/${train_set_sp}_hires $tgtdir/data/${train_set_sp}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 30 \
    $tgtdir/data/${train_set_sp}_max2_hires $tgtdir/exp/nnet3/extractor $tgtdir/exp/nnet3/ivectors_$train_set_sp || exit 1;

  for data_set in $test_set_1 $test_set_2; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 10 \
      $tgtdir/data/${data_set_sp}_hires $tgtdir/exp/nnet3/extractor $tgtdir/exp/nnet3/ivectors_$data_set_sp || exit 1;
  done
fi

exit 0;
