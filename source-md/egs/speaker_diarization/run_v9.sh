#!/bin/bash

# x-vector model:
# I use ellenrao x-vector model,its details: 
# here we don't train x-vector model, however, we use ellenrao's x-vector model
#(path:/home4/backup/ellenrao/spkver/nist18/exp/xvector_nnet_1a_wvox/)
# the model is retrain model.Input data is /home4/backup/ellenrao/spkver/nist18/data/swbd_sre_vox_combined_no_sil
#                            network structure is /home4/backup/ellenrao/spkver/nist18/local/nnet3/xvector/run_xvector.sh 

# plad model :
# sre_2004_2008 is used to train plda model, more details you can see it on run_v1.sh
# here sre data is sre_2004_2008 folder
# make sre data.
# because I don't have SRE2006 Test 2 ,so I use SRE2010 
# here sre data is as follows:
#     Corpus              LDC Catalog No.
#     SRE2004             LDC2006S44
#     SRE2005 Train       LDC2011S01
#     SRE2005 Test        LDC2011S04
#     SRE2006 Train       LDC2011S09
#     SRE2006 Test        LDC2011S10
#     SRE2008 Train       LDC2011S05
#     SRE2008 Test        LDC2011S08

# test data:
# it is from hangzhou  server. its path:/home/hhx502/w2019/projects/speaker_dz_test_nov02
# the test data only two wave files. I will named it speaker_dz_test_nov02
# so I can't score it.

# so v9 is as v8 but i used sre_2004_2008 data to train plad model.
. ./path.sh
. ./cmd.sh
set -e

steps=1


. parse_options.sh || exit 1
steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }} print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    declare step$index=1
  done
fi
# if we don't score it , kaldi format of the test data  requires: wav.scp segments  utt2dur utt2spk spk2utt reco2num_spk

# The segments file format is:
# <segment-id> <recording-id> <start-time> <end-time>
# The labels file format is:
# <segment-id> <speaker-id>

# rttm format is:
# <type> <file> <chnl> <tbeg> \
#        <tdur> <ortho> <stype> <name> <conf> <slat>
#where:
#<type> = "SPEAKER"
#<file> = <recording-id>
#<chnl> = "0"
#<tbeg> = start time of segment
#<tdur> = duration of segment
#<ortho> = "<NA>"
#<stype> = "<NA>"
#<name> = <speaker-id>
#<conf> = "<NA>"
#<slat> = "<NA>"

# 6. Extract x-vectors
sub_test_data_1=speaker_dz_test_nov02_1
sub_test_data_2=speaker_dz_test_nov02_2
nnet_dir=exp/ellerao_retrain_xvector_model
if [ ! -z $step06 ]; then
  # Extract x-vectors for the two partitions of test set.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 1 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_1}_cmn $nnet_dir/xvectors_${sub_test_data_1}

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 1 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_2}_cmn $nnet_dir/xvectors_${sub_test_data_2}
fi

sub_test_data_1=speaker_dz_test_nov02_1
sub_test_data_2=speaker_dz_test_nov02_2
nnet_dir=exp/ellerao_retrain_xvector_model
# 7. Train PLDA models
train_PLDA_set_name=sre_2004_2008
if [ ! -z $step07 ]; then
  # Train a PLDA model on $train_PLDA_set_name, using ${sub_test_data_1} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_1}_${train_PLDA_set_name}/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_1}/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_1}/plda || exit 1;

  # Train a PLDA model on $train_PLDA_set_name, using ${sub_test_data_2} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_2}_${train_PLDA_set_name}/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_2}/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_2}/plda || exit 1;
fi

# 8. Perform PLDA scoring
if [ ! -z $step08 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  # The first directory contains the PLDA model that used ${sub_test_data_2}
  # to perform whitening (recall that we're treating ${sub_test_data_2} as a
  # held-out dataset).  The second directory contains the x-vectors
  # for ${sub_test_data_1}.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 1 $nnet_dir/xvectors_${sub_test_data_2} $nnet_dir/xvectors_${sub_test_data_1} \
    $nnet_dir/xvectors_${sub_test_data_1}/plda_scores

  # Do the same thing for ${sub_test_data_2}.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 1 $nnet_dir/xvectors_${sub_test_data_1} $nnet_dir/xvectors_${sub_test_data_2} \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores
fi
if [ ! -z $step09 ];then
  for dataset in ${sub_test_data_1} ${sub_test_data_2}; do
    echo "clustering threshold for $dataset"
    best_der=100
    best_threshold=0
    #utils/filter_scp.pl -f 2 data/$dataset/wav.scp \
    #  ${test_data_dir}/fullref.rttm > data/$dataset/ref.rttm

    # The threshold is in terms of the log likelihood ratio provided by the
    # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
    # In the following loop, we evaluate the clustering on a heldout dataset
    # (${sub_test_data_1} is heldout for ${sub_test_data_2} and vice-versa) using some reasonable
    # thresholds for a well-calibrated system.
    for threshold in -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3; do
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
        --threshold $threshold $nnet_dir/xvectors_${dataset}/plda_scores \
        $nnet_dir/xvectors_${dataset}/plda_scores_t$threshold
    done
  done
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ ! -z $step10 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --reco2num-spk data/${sub_test_data_1}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_1}/plda_scores $nnet_dir/xvectors_${sub_test_data_1}/plda_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --reco2num-spk data/${sub_test_data_2}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores $nnet_dir/xvectors_${sub_test_data_2}/plda_scores_num_spk

fi

