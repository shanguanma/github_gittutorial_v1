#!/bin/bash

# x-vector model:
# I use ellenrao x-vector model,its details: 
# here we don't train x-vector model, however, we use ellenrao's x-vector model
#(path:/home4/backup/ellenrao/spkver/nist18/exp/xvector_nnet_1a_wvox/)
# the model is retrain model.Input data is /home4/backup/ellenrao/spkver/nist18/data/swbd_sre_vox_combined_no_sil
#                            network structure is /home4/backup/ellenrao/spkver/nist18/local/nnet3/xvector/run_xvector.sh 

# plda model:
# I  use  mandarin data(e.g: hkust-mandarin-tele-trans-ldc2005t32_train it is from haihua, its raw path:/data/users/hhx502/ldc-cts2016/hkust-mandarin-tele-trans-ldc2005t32/data/local/data/train/ ) to train PLDA model.

# test data:
# it is from hangzhou  server. its path:/home/hhx502/w2019/projects/speaker_dz_test_nov02
# the test data only two wave files. I will named it speaker_dz_test_nov02
# so I can't score it.

 
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

# 1. wav.scp
test_set=speaker_dz_test_nov02
test_set_dir=data/speaker_dz_test_nov02
if [ ! -z $step01 ];then
    [ -d $test_set_dir ] || mkdir -p $test_set_dir
    find data/speaker_dz_test_nov02_corpus  -name "*.wav" | sort > $test_set_dir/wav-list.txt
    cat  $test_set_dir/wav-list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";' | sort  > $test_set_dir/wav.scp   
fi
# 2. utt2spk and spk2utt
if [ ! -z $step02 ];then
   #  make utt2spk spk2utt
   cat $test_set_dir/wav.scp | awk '{print $1, $1;}' > $test_set_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $test_set_dir/utt2spk > $test_set_dir/spk2utt 

fi
# 3 . get utt2dur  and segments
if [ ! -z $step03 ];then
    echo "## LOG : utt2dur @ `date` in'$test_set_dir'"
    utils/data/get_utt2dur.sh --cmd "$train_cmd" --nj 1 $test_set_dir
    echo "## LOG : utt2du in '$test_set_dir' @ `date`"
    utils/data/get_segments_for_data.sh $test_set_dir > $test_set_dir/segments
    utils/fix_data_dir.sh $test_set_dir
fi



# 5 . I assumed per wave file only includes two speaker.
# I manual the file (e.g: reco2num_spk )

# so far, I have prepared kaldi format data for test set
# now kaldi format of the test set includes wav.scp utt2spk spk2utt utt2dur segments

# 6. make two subdataset 
sub_test_data_dir_1=${test_set_dir}_1
sub_test_data_dir_2=${test_set_dir}_2
nwav=$(expr $(wc -l <$test_set_dir/wav.scp) / 2 )
if [ ! -z $step04 ];then

   utils/copy_data_dir.sh $test_set_dir $sub_test_data_dir_1
   utils/copy_data_dir.sh $test_set_dir $sub_test_data_dir_2

   utils/shuffle_list.pl $test_set_dir/wav.scp | head -n $nwav \
   | utils/filter_scp.pl - $test_set_dir/wav.scp \
    > $sub_test_data_dir_1/wav.scp
   utils/fix_data_dir.sh $sub_test_data_dir_1
   #utils/filter_scp.pl --exclude $sub_test_data_dir_1/wav.scp \
   #$test_data_dir/wav.scp > $sub_test_data_dir_2/wav.scp
   utils/shuffle_list.pl $test_set_dir/wav.scp | tail -n $nwav \
   | utils/filter_scp.pl - $test_set_dir/wav.scp \
    > $sub_test_data_dir_2/wav.scp
   utils/fix_data_dir.sh $sub_test_data_dir_2
   utils/filter_scp.pl $sub_test_data_dir_1/wav.scp $test_set_dir/reco2num_spk \
    > $sub_test_data_dir_1/reco2num_spk
   utils/filter_scp.pl $sub_test_data_dir_2/wav.scp $test_set_dir/reco2num_spk \
    > $sub_test_data_dir_2/reco2num_spk

fi

# next I start to prepare feature for test set($test_set_dir,$sub_test_data_dir_1,$sub_test_data_dir_2)
# make featuer(e.g: mfcc(23 dim),vad,cmn ) from two sub test set 
# make feature(e.g:mfcc(23 dim)) from test set
if [ ! -z $step05 ];then
  #  splits test set  into two parts, called
  #  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  for name in  speaker_dz_test_nov02 speaker_dz_test_nov02_1 speaker_dz_test_nov02_2; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 1 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in  speaker_dz_test_nov02_1 speaker_dz_test_nov02_2 ; do
    sid/compute_vad_decision.sh --nj 1 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done
fi


# 5. apply cmn
if [ ! -z $step05 ];then
   for name in  speaker_dz_test_nov02_1 speaker_dz_test_nov02_2; do
    local/nnet3/xvector/prepare_feats.sh --nj 1 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done
fi


# 6. Extract x-vectors
sub_test_data_1=speaker_dz_test_nov02_1
sub_test_data_2=speaker_dz_test_nov02_2
nnet_dir=exp/ellerao_retrain_xvector_model
if [ ! -z $step06 ]; then
  # Extract x-vectors for the two partitions of test set.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 1 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_1}_cmn $nnet_dir/xvectors_${sub_test_data_1}_mandarin

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 1 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_2}_cmn $nnet_dir/xvectors_${sub_test_data_2}_mandarin
fi 

# 7. Train PLDA models
train_PLDA_set_name=hkust-mandarin-tele-trans-ldc2005t32_train
if [ ! -z $step07 ]; then
  # Train a PLDA model on $train_PLDA_set_name, using ${sub_test_data_1} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_1}_mandarin/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_1}_mandarin/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda || exit 1;

  # Train a PLDA model on $train_PLDA_set_name, using ${sub_test_data_2} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_2}_mandarin/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_2}_mandarin/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda || exit 1;
fi

# 8. Perform PLDA scoring
if [ ! -z $step08 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  # The first directory contains the PLDA model that used ${sub_test_data_2}
  # to perform whitening (recall that we're treating ${sub_test_data_2} as a
  # held-out dataset).  The second directory contains the x-vectors
  # for ${sub_test_data_1}.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 1 $nnet_dir/xvectors_${sub_test_data_2}_mandarin $nnet_dir/xvectors_${sub_test_data_1}_mandarin \
    $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores

  # Do the same thing for ${sub_test_data_2}.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 1 $nnet_dir/xvectors_${sub_test_data_1}_mandarin $nnet_dir/xvectors_${sub_test_data_2}_mandarin \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores
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
        --threshold $threshold $nnet_dir/xvectors_${dataset}_mandarin/plda_scores \
        $nnet_dir/xvectors_${dataset}_mandarin/plda_scores_t$threshold
    done
  done
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ ! -z $step10 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --reco2num-spk data/${sub_test_data_1}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --reco2num-spk data/${sub_test_data_2}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores_num_spk

fi
