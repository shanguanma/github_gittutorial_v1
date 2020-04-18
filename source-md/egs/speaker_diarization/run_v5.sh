#!/bin/bash



# v5 is same as v4, but I used full test wave file(it contains 300 wave files) as test set,then spilt the test set into two part on average.
# steps20-21 , i don't run it.
. ./cmd.sh
. ./path.sh

set -e

steps=1
#train_plda_data_dir=/home/ellenrao/spkver/nist18/data/sre_combined
#nnet_dir=exp/xvector_nnet_1a/
num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)


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
# Now I make kaldi data(it contains wav.scp segments utt2spk spk2utt fullref.rttm labels reco2num_spk) 
#  from textGrid format transcription and wave file .
# step 1 make wav.scp
if [ ! -z $step01 ];then
   # unzip the file
   if [ ! -d data/test-data-20190808/sd_hdu_cn ];then
     tar xzf data/test-data-20190808/sd_hdu_cn.tar.gz -C data/test-data-20190808/
   fi
   internal_folder=data/test-data-20190808-tmp
   mkdir -p $internal_folder
   # the follow command can get abosulte path in ./wav-list.txt
   # find /home3/md510/w2019a/kaldi-recipe/speaker_diarization/data/test-data-20190808/sd_hdu_cn -name "*.wav" | sort > ./wav-list.txt
   find data/test-data-20190808/sd_hdu_cn -name "*.wav" | sort > $internal_folder/wav-list.txt
   internal_folder_small=data/test-data-20190808-tmp_entire_v5
   mkdir -p $internal_folder_small
   head -n 300 $internal_folder/wav-list.txt > $internal_folder_small/wav-list.txt
   cat  $internal_folder_small/wav-list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";' | sort  > $internal_folder_small/wav.scp
   
fi
if [ ! -z $step02 ];then
  # add prefix (e.g:sd-hdu-cn) to recorder id 
  
  internal_folder_small=data/test-data-20190808-tmp_entire_v5
  data=data/test-data-20190808-kaldi_entire_v5
  mkdir -p $data
  cat $internal_folder_small/wav.scp | \
  source-md/egs/speaker_diarization/sge-manual-wavscp.pl sd-hdu-cn >  $data/wav.scp
  #source/egs/mandarin/update-lexicon-transcript-oct13/sge-manual-wavscp.pl sd-hdu-cn >  $data/wav.scp
fi


if [ ! -z $step03 ]; then
  internal_folder=data/test-data-20190808-tmp
  data=data/test-data-20190808-kaldi_entire_v5
  if [ ! -e $internal_folder/text-grid-list.txt ]; then
    find  data/test-data-20190808/sd_hdu_cn -name "*.TextGrid" | sort -u > $internal_folder/text-grid-list.txt
  fi
  internal_folder_small=data/test-data-20190808-tmp_entire_v5
  head -n 300 $internal_folder/text-grid-list.txt > $internal_folder_small/text-grid-list.txt
  cat $internal_folder_small/text-grid-list.txt | \
  source/egs/mandarin/update-lexicon-transcript-oct13/sge-manual-transcript.pl $data/wav.scp > $internal_folder_small/transcript.txt
  # remove silence duration,in other words, remove duration of no speaker info  
  # for example:$internal_folder_small/transcript.txt is as follow
  # sd-hdu-cn-0001-485995239007-spker-s001 sd-hdu-cn-0001-485995239007 0.00 39.96
  # sd-hdu-cn-0001-485995239007-spker-s001 sd-hdu-cn-0001-485995239007 39.96 40.32 S1m
  # .
  # .
  # .
  # we remove the line(e.g:sd-hdu-cn-0001-485995239007-non-non-spker-s001 sd-hdu-cn-0001-485995239007-non-non 0.00 39.96)
  awk '{if (NF==5){print $0}}' $internal_folder_small/transcript.txt>$internal_folder_small/transcript_new.txt
  # get utt2spk, spk2utt, segments,text file
  # their formats are as follows:
  # utt2spk: <segId> <wavid>
  # spk2utt: <wavid> <segId>
  # segments: <segId> <wavid> <startId> <endId>
  # text: <segid> <specify name of a speaker>
  # For clarity, I also interpred wav.scp format.
  # wav.scp : <wavid> <path of a wave file>
  # <segId> is <wavid-startId-endId>
  cat $internal_folder_small/transcript_new.txt | \
  source-md/egs/seame/mandarin/manual-kaldi-data.pl $data
fi

if [ ! -z $step04 ];then
  # make recod2spk file
  internal_folder_small=data/test-data-20190808-tmp_entire_v5
  data=data/test-data-20190808-kaldi_entire_v5
  awk '{print $2, $5 }' $internal_folder_small/transcript_new.txt > $internal_folder_small/transcript_pre_reco2spk.txt
  source-md/egs/speaker_diarization/make_reco2num_spk.py   $internal_folder_small/transcript_pre_reco2spk.txt  $data/reco2num_spk || exit 1;
fi

if [ ! -z $step05 ];then
   # make labels 
   data=data/test-data-20190808-kaldi_entire_v5
   source-md/egs/speaker_diarization/make_labels.py $data/text $data/labels
fi

if [ ! -z $step06 ];then
  rttm_channel=0
  # make fullref.rttm
  #The segments file format is:
  # <segment-id> <recording-id> <start-time> <end-time>
  #The labels file format is:
  #<segment-id> <speaker-id>

  #The output RTTM format is:
  #<type> <file> <chnl> <tbeg> \
  #      <tdur> <ortho> <stype> <name> <conf> <slat>
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
  echo "$0: prepare fullref.rttm"
  data=data/test-data-20190808-kaldi_entire_v5
  diarization/make_rttm.py --rttm-channel $rttm_channel $data/segments $data/labels $data/fullref.rttm || exit 1;
fi

# Next prepare the feature files
# make two subdataset 
test_data_dir=data/test-data-20190808-kaldi_entire_v5
sub_test_data_dir_1=${test_data_dir}_1
sub_test_data_dir_2=${test_data_dir}_2
nwav=$(expr $(wc -l <$test_data_dir/wav.scp) / 2 )


if [ ! -z $step08 ];then

   utils/copy_data_dir.sh $test_data_dir $sub_test_data_dir_1
   utils/copy_data_dir.sh $test_data_dir $sub_test_data_dir_2

   utils/shuffle_list.pl $test_data_dir/wav.scp | head -n $nwav \
   | utils/filter_scp.pl - $test_data_dir/wav.scp \
    > $sub_test_data_dir_1/wav.scp
   utils/fix_data_dir.sh $sub_test_data_dir_1
   #utils/filter_scp.pl --exclude $sub_test_data_dir_1/wav.scp \
   #$test_data_dir/wav.scp > $sub_test_data_dir_2/wav.scp
   utils/shuffle_list.pl $test_data_dir/wav.scp | tail -n $nwav \
   | utils/filter_scp.pl - $test_data_dir/wav.scp \
    > $sub_test_data_dir_2/wav.scp
   utils/fix_data_dir.sh $sub_test_data_dir_2
   utils/filter_scp.pl $sub_test_data_dir_1/wav.scp $test_data_dir/reco2num_spk \
    > $sub_test_data_dir_1/reco2num_spk
   utils/filter_scp.pl $sub_test_data_dir_2/wav.scp $test_data_dir/reco2num_spk \
    > $sub_test_data_dir_2/reco2num_spk

fi


# make feature(e.g:mfcc(23 dim), vad) from train set (e.g:swbd)
# train set is used to train gmm_ubm model in VB system.
if [ ! -z $step11 ]; then
  
  for name in swbd; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in swbd ; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
    
  done
fi

# make feature(e.g.:mfcc(23 dim),vad,cmn) from hkust mandarin (e.g:data/local/hkust-mandarin-tele-trans-ldc2005t32_train)
# data/local/hkust-mandarin-tele-trans-ldc2005t32_train is used to train plda model.
if [ ! -z $step12 ];then
  
  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.

  # using mandarin data(e.g.:data/local/hkust-mandarin-tele-trans-ldc2005t32_train) to train PLDA model
  # cp -r /data/users/hhx502/ldc-cts2016/hkust-mandarin-tele-trans-ldc2005t32/data/local/data/train data/local/ 
  # mv data/local/train data/local/hkust-mandarin-tele-trans-ldc2005t32_train
  # note:data/local/hkust-mandarin-tele-trans-ldc2005t32_train/segments contains
  # 210939_A-20040527_210939_a901153_b901154_A_35690_35690 20040527_210939_a901153_b901154_A 356.90 356.90
  # this is a badly segment, so it should be remove.
  # so I manual remove it, then  utils/fix_data_dir.sh  data/local/hkust-mandarin-tele-trans-ldc2005t32_train 
  for name in hkust-mandarin-tele-trans-ldc2005t32_train; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/local/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/local/$name
  done

  for name in hkust-mandarin-tele-trans-ldc2005t32_train ; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/local/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/local/$name

  done

  for name in hkust-mandarin-tele-trans-ldc2005t32_train ; do
    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
      data/local/$name data/local/${name}_cmn exp/${name}_cmn
    cp data/local/$name/vad.scp data/local/${name}_cmn/
    if [ -f data/local/$name/segments ]; then
      cp data/local/$name/segments data/local/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/local/${name}_cmn
  done
  echo "0.01" > data/local/hkust-mandarin-tele-trans-ldc2005t32_train_cmn/frame_shift
  # Create segments to extract x-vectors from for PLDA training data.
  # The segments are created using an energy-based speech activity
  # detection (SAD) system, but this is not necessary.  You can replace
  # this with segments computed from your favorite SAD.
  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
    data/local/hkust-mandarin-tele-trans-ldc2005t32_train_cmn data/local/hkust-mandarin-tele-trans-ldc2005t32_train_cmn_segmented
fi
# next I start to prepare feature for test set($test_data_dir,$sub_test_data_dir_1,$sub_test_data_dir_2)
# make featuer(e.g: mfcc(23 dim),vad,cmn ) from two sub test set 
# make feature(e.g:mfcc(23 dim)) from test set
if [ ! -z $step13 ];then
  #  splits test set  into two parts, called
  #  Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  for name in  test-data-20190808-kaldi_entire_v5 test-data-20190808-kaldi_entire_v5_1 test-data-20190808-kaldi_entire_v5_2; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in  test-data-20190808-kaldi_entire_v5_1 test-data-20190808-kaldi_entire_v5_2 ; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done
fi


if [ ! -z $step14 ];then
  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  for name in  test-data-20190808-kaldi_entire_v5_1 test-data-20190808-kaldi_entire_v5_2; do
    local/nnet3/xvector/prepare_feats.sh --nj 1 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done
fi
# Extract x-vectors
sub_test_data_1=test-data-20190808-kaldi_entire_v5_1
sub_test_data_2=test-data-20190808-kaldi_entire_v5_2
nnet_dir=exp/ellerao_retrain_xvector_model

if [ ! -z $step15 ]; then
  # Extract x-vectors for the two partitions of test-data-2019080-kaldi_demo_1.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 10 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_1}_cmn $nnet_dir/xvectors_${sub_test_data_1}_mandarin

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 10 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_2}_cmn $nnet_dir/xvectors_${sub_test_data_2}_mandarin

  # Reduce the amount of swbding data for the PLDA,
  utils/subset_data_dir.sh \
    data/local/hkust-mandarin-tele-trans-ldc2005t32_train_cmn_segmented 128000 data/local/hkust-mandarin-tele-trans-ldc2005t32_train_cmn_segmented_128k
  # Extract x-vectors for the hkust-mandarin, which is our PLDA swbding
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    --nj 30 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true $nnet_dir \
    data/local/hkust-mandarin-tele-trans-ldc2005t32_train_cmn_segmented_128k \
    $nnet_dir/xvectors_hkust-mandarin-tele-trans-ldc2005t32_train_segmented_128k
fi

# Train PLDA models
train_PLDA_set_name=hkust-mandarin-tele-trans-ldc2005t32_train
if [ ! -z $step16 ]; then
  # Train a PLDA model on SRE, using ${sub_test_data_1} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_1}_mandarin/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_1}_mandarin/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda || exit 1;

  # Train a PLDA model on SRE, using ${sub_test_data_2} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_2}_mandarin/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_${train_PLDA_set_name}_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_2}_mandarin/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda || exit 1;
fi

# Perform PLDA scoring
if [ ! -z $step17 ]; then
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

test_data=test-data-20190808-kaldi_entire_v5
# Cluster the PLDA scores using a stopping threshold.
if [ ! -z $step18 ]; then
  # First, we find the threshold that minimizes the DER on each partition of
  # $test_data_dir.
  mkdir -p $nnet_dir/tuning
  for dataset in ${sub_test_data_1} ${sub_test_data_2}; do
    echo "Tuning clustering threshold for $dataset"
    best_der=100
    best_threshold=0
    utils/filter_scp.pl -f 2 data/$dataset/wav.scp \
      ${test_data_dir}/fullref.rttm > data/$dataset/ref.rttm

    # The threshold is in terms of the log likelihood ratio provided by the
    # PLDA scores.  In a perfectly calibrated system, the threshold is 0.
    # In the following loop, we evaluate the clustering on a heldout dataset
    # (${sub_test_data_1} is heldout for ${sub_test_data_2} and vice-versa) using some reasonable
    # thresholds for a well-calibrated system.
    for threshold in -0.3 -0.2 -0.1 -0.05 0 0.05 0.1 0.2 0.3; do
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
        --threshold $threshold $nnet_dir/xvectors_${dataset}_mandarin/plda_scores \
        $nnet_dir/xvectors_${dataset}_mandarin/plda_scores_t$threshold

      md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm \
       -s $nnet_dir/xvectors_${dataset}_mandarin/plda_scores_t$threshold/rttm \
       2> $nnet_dir/tuning/${dataset}_mandarin_t${threshold}.log \
       > $nnet_dir/tuning/${dataset}_mandarin_t${threshold}

      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $nnet_dir/tuning/${dataset}_mandarin_t${threshold})
      if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
        best_der=$der
        best_threshold=$threshold
      fi
    done
    echo "$best_threshold" > $nnet_dir/tuning/${dataset}_mandarin_best
  done
  # Cluster ${sub_test_data_1} using the best threshold found for ${sub_test_data_2}.  This way,
  # ${sub_test_data_2} is treated as a held-out dataset to discover a reasonable
  # stopping threshold for ${sub_test_data_1}.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --threshold $(cat $nnet_dir/tuning/${sub_test_data_2}_mandarin_best) \
    $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores

  # Do the same thing for ${sub_test_data_2}, treating ${sub_test_data_1} as a held-out dataset
  # to discover a stopping threshold.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --threshold $(cat $nnet_dir/tuning/${sub_test_data_1}_mandarin_best) \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores

  mkdir -p $nnet_dir/results_${test_data}_mandarin_supervised
  # Now combine the results for ${sub_test_data_1} and ${sub_test_data_1} and evaluate it
  # together.
  cat $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores/rttm \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    ${test_data_dir}/fullref.rttm -s - 2> $nnet_dir/results_${test_data}_mandarin_supervised/threshold.log \
    > $nnet_dir/results_${test_data}_mandarin_supervised/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results_${test_data}_mandarin_supervised/DER_threshold.txt)
  # Using supervised calibration, DER:10.84
  # path:cat exp/ellerao_retrain_xvector_model/results_test-data-20190808-kaldi_entire_v5_mandarin_supervised/DER_threshold.txt

  echo "Using supervised calibration, DER: $der%"
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ ! -z $step19 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --reco2num-spk data/${sub_test_data_1}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 1 \
    --reco2num-spk data/${sub_test_data_2}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores_num_spk

  mkdir -p $nnet_dir/results_${test_data}_mandarin_oracle
  # Now combine the results for ${sub_test_data_1} and ${sub_test_data_2} and evaluate it together.
  cat $nnet_dir/xvectors_${sub_test_data_1}_mandarin/plda_scores_num_spk/rttm \
  $nnet_dir/xvectors_${sub_test_data_2}_mandarin/plda_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r ${test_data_dir}/fullref.rttm -s - 2> $nnet_dir/results_${test_data}_mandarin_oracle/num_spk.log \
    > $nnet_dir/results_${test_data}_mandarin_oracle/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results_${test_data}_mandarin_oracle/DER_num_spk.txt)
  # Using the oracle number of speakers, DER:21.84%
  # cat exp/ellerao_retrain_xvector_model/results_test-data-20190808-kaldi_entire_v5_mandarin_oracle/DER_num_spk.txt
  echo "Using the oracle number of speakers, DER: $der%"
fi

# Variational Bayes resegmentation using the code from Brno University of Technology
# Please see https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors 
# for details
# VB system is still fail. 
# error log:exp/VB_20190808_demo_1/log/VB_resegmentation.1.log
# I am checking.
if [ ! -z $step20 ]; then
  #utils/subset_data_dir.sh data/swbd 32000 data/train_32k
  # Train the diagonal UBM.
  sid/train_diag_ubm.sh --cmd "$train_cmd --mem 8G" \
    --nj 40 --num-threads 8 --subsample 1 --delta-order 0 --apply-cmn false \
    data/swbd $num_components exp/diag_ubm_$num_components

  # Train the i-vector extractor. The UBM is assumed to be diagonal.
  diarization/train_ivector_extractor_diag.sh \
    --cmd "$train_cmd --mem 8G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 40 \
    exp/diag_ubm_$num_components/final.dubm data/swbd \
    exp/extractor_diag_c${num_components}_i${ivector_dim}
fi

if [ ! -z $step21 ]; then
  VB_name=VB_20190808_demo_1
  output_rttm_dir=exp/$VB_name/rttm
  mkdir -p $output_rttm_dir || exit 1;
  cat $nnet_dir/xvectors_${sub_test_data_1}/plda_scores/rttm \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores/rttm > $output_rttm_dir/x_vector_rttm
  init_rttm_file=$output_rttm_dir/x_vector_rttm

  # VB resegmentation. In this script, I use the x-vector result to 
  # initialize the VB system. You can also use i-vector result or random 
  # initize the VB system. The following script uses kaldi_io. 
  # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
  diarization/VB_resegmentation.sh --nj 1 --cmd "$train_cmd --mem 10G" \
    --initialize 1 ${test_data_dir}  $init_rttm_file exp/$VB_name \
    exp/diag_ubm_$num_components/final.dubm exp/extractor_diag_c${num_components}_i${ivector_dim}/final.ie || exit 1;

  # Compute the DER after VB resegmentation
  mkdir -p exp/$VB_name/results || exit 1;
  md-eval.pl -1 -c 0.25 -r ${test_data_dir}/fullref.rttm -s $output_rttm_dir/VB_rttm 2> exp/$VB_name/log/VB_DER.log \
    > exp/$VB_name/results/VB_DER.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/$VB_name/results/VB_DER.txt)
  # After VB resegmentation, DER:
  echo "After VB resegmentation, DER: $der%"
fi
                                   
