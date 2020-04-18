#!/bin/bash

# the script is modified from kaldi/egs/callhome_diarization/v2/run.sh
# we use the script (e.g :/home/ellenrao/spkver/nist18/scripts/run_newxvector_wvox.sh )
# to generat the xvector model.
# the script is just only used to familiraize with the speaker diarization process.


# The overall idea of this script is as follows:
# use ellenral's x-vector model,
# method1:use sre dataset to train plda model with callhome1 or callhome2 as whiten;
# use train set (e.g. it contain swbd dataset and sre dataset) to train Variational Bayes resegmentation systems to impove speaker diarization.

# swbd dataset contains  /home/ellenrao/spkver/nist18/(data/swbd_cellular1_train, data/swbd_cellular2_train,data/swbd2_phase1_train, data/swbd2_phase2_train, data/swbd2_phase3_train)
# sre dataset contains /home/ellenrao/spkver/nist18/(data/sre2004, data/sre2005_train, data/sre2005_test, data/sre2006_train,
# data/sre2006_test, data/sre08, data/mx6, data/sre10)
# more details :you can read the script(/home/ellenrao/spkver/nist18/scripts/run_newxvector_wvox.sh) 

# current folder path:/home3/md510/w2019a/kaldi-recipe/speaker_diarization

# again test speaker diarization from data/test-data-20190808/sd_hdu_cn.tar.gz

# Let me describe the details of the data. its wave file is 8000Hz single channel 16bit.
# speaker details is stored at textGrid file corresponding to wav file.

. ./cmd.sh
. ./path.sh

set -e

steps=1
#train_plda_data_dir=/home/ellenrao/spkver/nist18/data/sre_combined
nnet_dir=exp/xvector_nnet_1a/
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
   if [ -d data/test-data-20190808/sd_hdu_cn ];then
     tar xzf data/test-data-20190808/sd_hdu_cn.tar.gz -C data/test-data-20190808/
   fi
   internal_folder=data/test-data-2019080-tmp
   mkdir -p $internal_folder
   find data/test-data-20190808/sd_hdu_cn -name "*.wav" > $internal_folder/wav-list.txt
   cat  $internal_folder/wav-list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\_right.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";' | sort  > $internal_folder/wav.scp
   #cat $new_corpus_dir/wav.scp | sort >$new_corpus_dir/wav_1.scp
   #mv $new_corpus_dir/wav_1.scp $new_corpus_dir/wav.scp
fi

if [ ! -z $step02 ];then
  # add prefix (e.g:sd-hdu-cn) to recorder id 
  internal_folder=data/test-data-2019080-tmp
  data=data/test-data-2019080-kaldi
  mkdir -p $data
  cat $internal_folder/wav.scp | \
  source/egs/mandarin/update-lexicon-transcript-oct13/sge-manual-wavscp.pl sd-hdu-cn >  $data/wav.scp
fi

internal_folder=data/test-data-2019080-tmp
data=data/test-data-2019080-kaldi
if [ ! -z $step03 ]; then
  if [ ! -e $internal_folder/text-grid-list.txt ]; then
    find  data/test-data-20190808/sd_hdu_cn -name "*.TextGrid" | sort -u > $internal_folder/text-grid-list.txt
  fi
  cat $internal_folder/text-grid-list.txt | \
  source/egs/mandarin/update-lexicon-transcript-oct13/sge-manual-transcript.pl $data/wav.scp > $internal_folder/transcript.txt
  # remove silence duration,in other words, remove duration of no speaker info  
  # for example:$internal_folder/transcript.txt is as follow
  # sd-hdu-cn-0001-485995239007-non-non-spker-s001 sd-hdu-cn-0001-485995239007-non-non 0.00 39.96
  # sd-hdu-cn-0001-485995239007-non-non-spker-s001 sd-hdu-cn-0001-485995239007-non-non 39.96 40.32 S1m
  # .
  # .
  # .
  # we remove the line(e.g:sd-hdu-cn-0001-485995239007-non-non-spker-s001 sd-hdu-cn-0001-485995239007-non-non 0.00 39.96)
  awk '{if (NF==5){print $0}}' $internal_folder/transcript.txt>$internal_folder/transcript_new.txt
  # get utt2spk, spk2utt, segments,text file
  # their formats are as follows:
  # utt2spk: <segId> <wavid>
  # spk2utt: <wavid> <segId>
  # segments: <segId> <wavid> <startId> <endId>
  # text: <segid> <specify name of a speaker>
  # For clarity, I also interpred wav.scp format.
  # wav.scp : <wavid> <path of a wave file>
  # <segId> is <wavid-startId-endId>
  cat $internal_folder/transcript_new.txt | \
  source-md/egs/seame/mandarin/manual-kaldi-data.pl $data
fi

if [ ! -z $step04 ];then
  # make recod2spk file
  awk '{print $2, $5 }' $internal_folder/transcript_new.txt > $internal_folder/transcript_pre_reco2spk.txt
  source-md/egs/speaker_diarization/make_reco2num_spk.py   $internal_folder/transcript_pre_reco2spk.txt  $data/reco2num_spk || exit 1;
fi

if [ ! -z $step05 ];then
   # make labels 
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
  diarization/make_rttm.py --rttm-channel $rttm_channel $data/segments $data/labels $data/fullref.rttm || exit 1;
fi


# All the files(e.g.wav.scp utt2spk spk2utt segments labels reco2num_spk fullref.rttm) kaldi speaker diarization needs are ready

# 
# Next prepare the feature files
# make two subdataset 
test_data_dir=data/test-data-2019080-kaldi
sub_test_data_dir_1=${test_data_dir}_1
sub_test_data_dir_2=${test_data_dir}_2
nwav=$(expr $(wc -l <$test_data_dir/wav.scp) / 2 )
cleanup=true
# remove some files 
# Every time I want to re-run this system, 
# remember to delete the generated features files and 
# regenerate them so that new and old files are not checked.
if [ ! -z $step07 ];then
   if [ -d $sub_test_data_dir_1 ];then
     rm -r $sub_test_data_dir_1
   fi

   if [ -d $sub_test_data_dir_2 ];then
     rm -r $sub_test_data_dir_2
   fi

   if [ -d ${sub_test_data_dir_1}_cmn ] ;then 
     rm -r ${sub_test_data_dir_1}_cmn
   fi

   if [ -d ${sub_test_data_dir_1}_cmn ];then
     rm -r ${sub_test_data_dir_1}_cmn
   fi    
         
   rm -r $test_data_dir/conf $test_data_dir/data \
         $test_data_dir/feats.scp $test_data_dir/frame_shift \
         $test_data_dir/utt2dur $test_data_dir/utt2num_frames 
fi 

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
# Prepare features
# note: beacuse I used ellenrao's xvector model(e.g:exp/ellerao_retrain_xvector_model),
# so I did't prepare train set feature and training xvetor model process.
# # here we don't train x-vector model, however, we use ellenrao's x-vector model
  #(path:/home/ellenrao/spkver/nist18/exp/xvector_nnet_1a_wvox/)
  # the model is retrain model.Input data is /home/ellenrao/spkver/nist18/data/swbd_sre_vox_combined_no_sil
  #                            network structure is /home/ellenrao/spkver/nist18/local/nnet3/xvector/run_xvector.sh 


# strictly follow callhome_diarization configure.
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

if [ ! -z $step09 ];then
   # This prepares the  NIST SREs from 2004-2008.
   data_root=/data/users/ellenrao
   
  local/make_sre.sh $data_root/NIST_SRE_Corpus data/ 
fi 

# make train set for x-vector
if [ ! -z $step10 ];then
   # Prepare SWBD corpora.
  local/make_swbd_cellular1.pl /data/users/ellenrao/NIST_SRE_Corpus/switchboard/Switchboard-1-LDC2001S13 \
    data/swbd_cellular1_train
  local/make_swbd_cellular2.pl /data/users/ellenrao/NIST_SRE_Corpus/switchboard/Switchboard-Cell-P2-LDC2004S07 \
    data/swbd_cellular2_train
  local/make_swbd2_phase1.pl /data/users/ellenrao/NIST_SRE_Corpus/switchboard/Switchboard-2P1-LDC98S75 \
    data/swbd2_phase1_train
  local/make_swbd2_phase2.pl /data/users/ellenrao/NIST_SRE_Corpus/switchboard/Switchboard-2P2-LDC99S79 \
    data/swbd2_phase2_train
  local/make_swbd2_phase3.pl /data/users/ellenrao/NIST_SRE_Corpus/switchboard/Switchboard-2P3-LDC2002S06 \
    data/swbd2_phase3_train

  # Combine all SWB corpora into one dataset.
  utils/combine_data.sh data/swbd \
    data/swbd_cellular1_train data/swbd_cellular2_train \
    data/swbd2_phase1_train data/swbd2_phase2_train data/swbd2_phase3_train

fi 


if [ ! -z $step11 ]; then
  # split test-data-2019080-kaldi into two parts, called
  # test-data-2019080-kaldi_1 and test-data-2019080-kaldi_2.  
  # Each partition is treated like a held-out
  # dataset, and used to estimate various quantities needed to perform
  # diarization on the other part (and vice versa).
  
  for name in swbd  test-data-2019080-kaldi_1  test-data-2019080-kaldi_2 test-data-2019080-kaldi; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in swbd  test-data-2019080-kaldi_1  test-data-2019080-kaldi_2; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name  
    # error log:
    # fix_data_dir.sh: kept all 2050 utterances.
    #fix_data_dir.sh: old files are kept in data/test-data-2019080-kaldi_2/.backup
    # fix_data_dir.sh: no utterances remained: not proceeding further.

  done
fi

if [ ! -z $step12 ];then
   # The sre dataset is a subset of train
  cp data/swbd/{feats,vad}.scp data/sre_2004_2008/
  utils/fix_data_dir.sh data/sre_2004_2008
  # This writes features to disk after applying the sliding window CMN.
  # Although this is somewhat wasteful in terms of disk space, for diarization
  # it ends up being preferable to performing the CMN in memory.  If the CMN
  # were performed in memory (e.g., we used --apply-cmn true in
  # diarization/nnet3/xvector/extract_xvectors.sh) it would need to be
  # performed after the subsegmentation, which leads to poorer results.
  for name in sre_2004_2008; do
    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

  echo "0.01" > data/sre_2004_2008_cmn/frame_shift
  # Create segments to extract x-vectors from for PLDA training data.
  # The segments are created using an energy-based speech activity
  # detection (SAD) system, but this is not necessary.  You can replace
  # this with segments computed from your favorite SAD.
  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
    data/sre_2004_2008_cmn data/sre_2004_2008_cmn_segmented
fi

#if [ ! -z $step13 ];then
  # make vad feature
#  for name in  test-data-2019080-kaldi_1  test-data-2019080-kaldi_2; do
#    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
#      data/$name exp/make_vad $vaddir
#    utils/fix_data_dir.sh data/$name
#  done

#fi

if [ ! -z $step13 ];then
  for name in  test-data-2019080-kaldi_1  test-data-2019080-kaldi_2; do
    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
  done

fi

# In this section, we augment the training data with reverberation,
# noise, music, and babble, and combined it with the clean data.
# The combined list will be used to train the xvector DNN.  The SRE
# subset will be used to train the PLDA model.
if [ ! -z $step14 ]; then
  frame_shift=0.01
  awk -v frame_shift=$frame_shift '{print $1, $2*frame_shift;}' data/swbd/utt2num_frames > data/swbd/reco2dur
  if [ ! -d "RIRS_NOISES" ]; then
    # Download the package that includes the real RIRs, simulated RIRs, isotropic noises and point-source noises
    wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
    unzip rirs_noises.zip
  fi

  # Make a version with reverberated speech
  rvb_opts=()
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/smallroom/rir_list")
  rvb_opts+=(--rir-set-parameters "0.5, RIRS_NOISES/simulated_rirs/mediumroom/rir_list")

  # Make a reverberated version of the SWBD+SRE list.  Note that we don't add any
  # additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications 1 \
    --source-sampling-rate 8000 \
    data/swbd data/swbd_reverb
  cp data/swbd/vad.scp data/swbd_reverb/
  utils/copy_data_dir.sh --utt-suffix "-reverb" data/swbd_reverb data/swbd_reverb.new
  rm -rf data/swbd_reverb
  mv data/swbd_reverb.new data/swbd_reverb

  # Prepare the MUSAN corpus, which consists of music, speech, and noise
  # suitable for augmentation.
  steps/data/make_musan.sh --sampling-rate 8000 /data/users/ellenrao/musan data

  # Get the duration of the MUSAN recordings.  This will be used by the
  # script augment_data_dir.py.
  for name in speech noise music; do
    utils/data/get_utt2dur.sh data/musan_${name}
    mv data/musan_${name}/utt2dur data/musan_${name}/reco2dur
  done

  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-suffix "noise" --fg-interval 1 --fg-snrs "15:10:5:0" --fg-noise-dir "data/musan_noise" data/swbd data/swbd_noise
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-suffix "music" --bg-snrs "15:10:8:5" --num-bg-noises "1" --bg-noise-dir "data/musan_music" data/swbd data/swbd_music
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-suffix "babble" --bg-snrs "20:17:15:13" --num-bg-noises "3:4:5:6:7" --bg-noise-dir "data/musan_speech" data/swbd data/swbd_babble

  # Combine reverb, noise, music, and babble into one directory.
  utils/combine_data.sh data/swbd_aug data/swbd_reverb data/swbd_noise data/swbd_music data/swbd_babble

  # Take a random subset of the augmentations (128k is somewhat larger than twice
  # the size of the SWBD+SRE list)
  utils/subset_data_dir.sh data/swbd_aug 128000 data/swbd_aug_128k
  utils/fix_data_dir.sh data/swbd_aug_128k

  # Make filterbanks for the augmented data.  Note that we do not compute a new
  # vad.scp file here.  Instead, we use the vad.scp from the clean version of
  # the list.
  steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 40 --cmd "$train_cmd" \
    data/swbd_aug_128k exp/make_mfcc $mfccdir

  # Combine the clean and augmented SWBD+SRE list.  This is now roughly
  # double the size of the original clean list.
  utils/combine_data.sh data/swbd_combined data/swbd_aug_128k data/swbd
fi

# Now we prepare the features to generate examples for xvector training.
if [ ! -z $step15 ]; then
  # This script applies CMN and removes nonspeech frames.  Note that this is somewhat
  # wasteful, as it roughly doubles the amount of training data on disk.  After
  # creating training examples, this can be removed.
  local/nnet3/xvector/prepare_feats_for_egs.sh --nj 40 --cmd "$train_cmd" \
    data/swbd_combined data/swbd_combined_cmn_no_sil exp/swbd_combined_cmn_no_sil
  utils/fix_data_dir.sh data/swbd_combined_cmn_no_sil

  # Now, we need to remove features that are too short after removing silence
  # frames.  We want atleast 5s (500 frames) per utterance.
  min_len=500
  mv data/swbd_combined_cmn_no_sil/utt2num_frames data/swbd_combined_cmn_no_sil/utt2num_frames.bak
  awk -v min_len=${min_len} '$2 > min_len {print $1, $2}' data/swbd_combined_cmn_no_sil/utt2num_frames.bak > data/swbd_combined_cmn_no_sil/utt2num_frames
  utils/filter_scp.pl data/swbd_combined_cmn_no_sil/utt2num_frames data/swbd_combined_cmn_no_sil/utt2spk > data/swbd_combined_cmn_no_sil/utt2spk.new
  mv data/swbd_combined_cmn_no_sil/utt2spk.new data/swbd_combined_cmn_no_sil/utt2spk
  utils/fix_data_dir.sh data/swbd_combined_cmn_no_sil

  # We also want several utterances per speaker. Now we'll throw out speakers
  # with fewer than 8 utterances.
  min_num_utts=8
  awk '{print $1, NF-1}' data/swbd_combined_cmn_no_sil/spk2utt > data/swbd_combined_cmn_no_sil/spk2num
  awk -v min_num_utts=${min_num_utts} '$2 >= min_num_utts {print $1, $2}' \
    data/swbd_combined_cmn_no_sil/spk2num | utils/filter_scp.pl - data/swbd_combined_cmn_no_sil/spk2utt \
    > data/swbd_combined_cmn_no_sil/spk2utt.new
  mv data/swbd_combined_cmn_no_sil/spk2utt.new data/swbd_combined_cmn_no_sil/spk2utt
  utils/spk2utt_to_utt2spk.pl data/swbd_combined_cmn_no_sil/spk2utt > data/swbd_combined_cmn_no_sil/utt2spk

  utils/filter_scp.pl data/swbd_combined_cmn_no_sil/utt2spk data/swbd_combined_cmn_no_sil/utt2num_frames > data/swbd_combined_cmn_no_sil/utt2num_frames.new
  mv data/swbd_combined_cmn_no_sil/utt2num_frames.new data/swbd_combined_cmn_no_sil/utt2num_frames

  # Now we're ready to create training examples.
  utils/fix_data_dir.sh data/swbd_combined_cmn_no_sil
  # train x-vector model
  local/nnet3/xvector/tuning/run_xvector_1a.sh --stage $stage --train-stage -1 \
  --data data/swbd_combined_cmn_no_sil --nnet-dir $nnet_dir \
  --egs-dir $nnet_dir/egs

fi





# Extract x-vectors
sub_test_data_1=test-data-2019080-kaldi_1
sub_test_data_2=test-data-2019080-kaldi_2
net_dir=exp/xvector_nnet_1a/
if [ ! -z $step16 ]; then
  # Extract x-vectors for the two partitions of test-data-2019080-kaldi.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_1}_cmn $nnet_dir/xvectors_${sub_test_data_1}

  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 5G" \
    --nj 40 --window 1.5 --period 0.75 --apply-cmn false \
    --min-segment 0.5 $nnet_dir \
    data/${sub_test_data_2}_cmn $nnet_dir/xvectors_${sub_test_data_2}

  # Reduce the amount of swbding data for the PLDA,
  utils/subset_data_dir.sh data/sre_2004_2008_cmn_segmented 128000 data/sre_2004_2008_cmn_segmented_128k
  # Extract x-vectors for the SRE, which is our PLDA swbding
  # data.  A long period is used here so that we don't compute too
  # many x-vectors for each recording.
  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
    --hard-min true $nnet_dir \
    data/sre_2004_2008_cmn_segmented_128k $nnet_dir/xvectors_sre_2004_2008_segmented_128k
fi

# Train PLDA models
if [ ! -z $step16 ]; then
  # Train a PLDA model on SRE, using ${sub_test_data_1} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_1}/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_sre_2004_2008_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_sre_2004_2008_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_1}/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_1}/plda || exit 1;

  # Train a PLDA model on SRE, using ${sub_test_data_2} to whiten.
  # We will later use this to score x-vectors in ${sub_test_data_1}.
  $train_cmd $nnet_dir/xvectors_${sub_test_data_2}/log/plda.log \
    ivector-compute-plda ark:$nnet_dir/xvectors_sre_2004_2008_segmented_128k/spk2utt \
      "ark:ivector-subtract-global-mean \
      scp:$nnet_dir/xvectors_sre_2004_2008_segmented_128k/xvector.scp ark:- \
      | transform-vec $nnet_dir/xvectors_${sub_test_data_2}/transform.mat ark:- ark:- \
      | ivector-normalize-length ark:- ark:- |" \
    $nnet_dir/xvectors_${sub_test_data_2}/plda || exit 1;
fi

# Perform PLDA scoring
if [ ! -z $step17 ]; then
  # Perform PLDA scoring on all pairs of segments for each recording.
  # The first directory contains the PLDA model that used ${sub_test_data_2}
  # to perform whitening (recall that we're treating ${sub_test_data_2} as a
  # held-out dataset).  The second directory contains the x-vectors
  # for ${sub_test_data_1}.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 $nnet_dir/xvectors_${sub_test_data_2} $nnet_dir/xvectors_${sub_test_data_1} \
    $nnet_dir/xvectors_${sub_test_data_1}/plda_scores

  # Do the same thing for ${sub_test_data_2}.
  diarization/nnet3/xvector/score_plda.sh --cmd "$train_cmd --mem 4G" \
    --nj 20 $nnet_dir/xvectors_${sub_test_data_1} $nnet_dir/xvectors_${sub_test_data_2} \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores
fi

test_data=test-data-2019080-kaldi
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
      diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
        --threshold $threshold $nnet_dir/xvectors_$dataset/plda_scores \
        $nnet_dir/xvectors_$dataset/plda_scores_t$threshold

      md-eval.pl -1 -c 0.25 -r data/$dataset/ref.rttm \
       -s $nnet_dir/xvectors_$dataset/plda_scores_t$threshold/rttm \
       2> $nnet_dir/tuning/${dataset}_t${threshold}.log \
       > $nnet_dir/tuning/${dataset}_t${threshold}

      der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
        $nnet_dir/tuning/${dataset}_t${threshold})
      if [ $(perl -e "print ($der < $best_der ? 1 : 0);") -eq 1 ]; then
        best_der=$der
        best_threshold=$threshold
      fi
    done
    echo "$best_threshold" > $nnet_dir/tuning/${dataset}_best
  done

  # Cluster ${sub_test_data_1} using the best threshold found for ${sub_test_data_2}.  This way,
  # ${sub_test_data_2} is treated as a held-out dataset to discover a reasonable
  # stopping threshold for ${sub_test_data_1}.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat $nnet_dir/tuning/${sub_test_data_2}_best) \
    $nnet_dir/xvectors_${sub_test_data_1}/plda_scores $nnet_dir/xvectors_${sub_test_data_1}/plda_scores

  # Do the same thing for ${sub_test_data_2}, treating ${sub_test_data_1} as a held-out dataset
  # to discover a stopping threshold.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" --nj 20 \
    --threshold $(cat $nnet_dir/tuning/${sub_test_data_1}_best) \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores $nnet_dir/xvectors_${sub_test_data_2}/plda_scores

  mkdir -p $nnet_dir/results_${test_data}_supervised
  # Now combine the results for ${sub_test_data_1} and ${sub_test_data_1} and evaluate it
  # together.
  cat $nnet_dir/xvectors_${sub_test_data_1}/plda_scores/rttm \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores/rttm | md-eval.pl -1 -c 0.25 -r \
    ${test_data_dir}/fullref.rttm -s - 2> $nnet_dir/results_${test_data}_supervised/threshold.log \
    > $nnet_dir/results_${test_data}_supervised/DER_threshold.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results_${test_data}_supervised/DER_threshold.txt)
  # Using supervised calibration, DER:6.45% 
 
  echo "Using supervised calibration, DER: $der%"
fi

# Cluster the PLDA scores using the oracle number of speakers
if [ ! -z $step19 ]; then
  # In this section, we show how to do the clustering if the number of speakers
  # (and therefore, the number of clusters) per recording is known in advance.
  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/${sub_test_data_1}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_1}/plda_scores $nnet_dir/xvectors_${sub_test_data_1}/plda_scores_num_spk

  diarization/cluster.sh --cmd "$train_cmd --mem 4G" \
    --reco2num-spk data/${sub_test_data_2}/reco2num_spk \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores $nnet_dir/xvectors_${sub_test_data_2}/plda_scores_num_spk

  mkdir -p $nnet_dir/results_${test_data}_oracle
  # Now combine the results for ${sub_test_data_1} and ${sub_test_data_2} and evaluate it together.
  cat $nnet_dir/xvectors_${sub_test_data_1}/plda_scores_num_spk/rttm \
  $nnet_dir/xvectors_${sub_test_data_2}/plda_scores_num_spk/rttm \
    | md-eval.pl -1 -c 0.25 -r ${test_data_dir}/fullref.rttm -s - 2> $nnet_dir/results_${test_data}_oracle/num_spk.log \
    > $nnet_dir/results_${test_data}_oracle/DER_num_spk.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    $nnet_dir/results_${test_data}_oracle/DER_num_spk.txt)
  # Using the oracle number of speakers, DER:
  echo "Using the oracle number of speakers, DER: $der%"
fi

# Variational Bayes resegmentation using the code from Brno University of Technology
# Please see https://speech.fit.vutbr.cz/software/vb-diarization-eigenvoice-and-hmm-priors 
# for details
if [ ! -z $step20 ]; then
  utils/subset_data_dir.sh data/swbd 32000 data/train_32k
  # Train the diagonal UBM.
  sid/swbd_diag_ubm.sh --cmd "$train_cmd --mem 8G" \
    --nj 40 --num-threads 8 --subsample 1 --delta-order 0 --apply-cmn false \
    data/swbd_32k $num_components exp/diag_ubm_$num_components

  # Train the i-vector extractor. The UBM is assumed to be diagonal.
  diarization/swbd_ivector_extractor_diag.sh \
    --cmd "$train_cmd --mem 8G" \
    --ivector-dim $ivector_dim --num-iters 5 --apply-cmn false \
    --num-threads 1 --num-processes 1 --nj 40 \
    exp/diag_ubm_$num_components/final.dubm data/swbd \
    exp/extractor_diag_c${num_components}_i${ivector_dim}
fi

if [ ! -z $step200 ]; then
  output_rttm_dir=exp/VB_20190808/rttm
  mkdir -p $output_rttm_dir || exit 1;
  cat $nnet_dir/xvectors_${sub_test_data_1}/plda_scores/rttm \
    $nnet_dir/xvectors_${sub_test_data_2}/plda_scores/rttm > $output_rttm_dir/x_vector_rttm
  init_rttm_file=$output_rttm_dir/x_vector_rttm

  # VB resegmentation. In this script, I use the x-vector result to 
  # initialize the VB system. You can also use i-vector result or random 
  # initize the VB system. The following script uses kaldi_io. 
  # You could use `sh ../../../tools/extras/install_kaldi_io.sh` to install it
  diarization/VB_resegmentation.sh --nj 20 --cmd "$train_cmd --mem 10G" \
    --initialize 1 ${test_data_dir}  $init_rttm_file exp/VB_20190808 \
    exp/diag_ubm_$num_components/final.dubm exp/extractor_diag_c${num_components}_i${ivector_dim}/final.ie || exit 1; 

  # Compute the DER after VB resegmentation
  mkdir -p exp/VB_20190808/results || exit 1;
  md-eval.pl -1 -c 0.25 -r ${test_data_dir}/fullref.rttm -s $output_rttm_dir/VB_rttm 2> exp/VB_20190808/log/VB_DER.log \
    > exp/VB_2019080/results/VB_DER.txt
  der=$(grep -oP 'DIARIZATION\ ERROR\ =\ \K[0-9]+([.][0-9]+)?' \
    exp/VB_20190808/results/VB_DER.txt)
  # After VB resegmentation, DER:
  echo "After VB resegmentation, DER: $der%"
fi
