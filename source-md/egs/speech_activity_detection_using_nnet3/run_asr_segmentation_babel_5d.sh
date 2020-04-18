#!/bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0

# refrence : /kaldi/egs/babel/s5d/local/run_asr_segmentation.sh
 
lang=data/lang   # Must match the one used to train the models
lang_test=data/local/lang_test  # Lang directory for decoding.

data_dir=data/train 
# Model directory used to align the $data_dir to get target labels for training
# SAD. This should typically be a speaker-adapted system.
sat_model_dir=exp/tri4
# Model direcotry used to decode the whole-recording version of the $data_dir to
# get target labels for training SAD. This should typically be a 
# speaker-independent system like LDA+MLLT system.
model_dir=exp/tri3
graph_dir=    # If not provided, a new one will be created using $lang_test

# Uniform segmentation options for decoding whole recordings. All values are in
# seconds.
#max_segment_duration=10
#overlap_duration=2.5
#max_remaining_duration=5  # If the last remaining piece when splitting uniformly
                          # is smaller than this duration, then the last piece 
                          # is  merged with the previous.

# List of weights on labels obtained from alignment, 
# labels obtained from decoding and default labels in out-of-segment regions
merge_weights=1.0,0.1,0.5

prepare_targets_stage=1-16
nstage=-10
train_stage=-10

affix=_1b
#stage=-1
steps=1
nj=80
reco_nj=40

exp_root= # project root folder, in this folder , it stored tri model.
# test options
test_stage=-10
test_nj=10
test_set="dev_man dev_sge"

# Babel specific configuration. These two lines can be removed when adapting to other corpora.
#[ ! -f ./lang.conf ] && echo 'Language configuration does not exist! Use the configurations in conf/lang/* as a startup' && exit 1
#. ./lang.conf || exit 1;

. ./path.sh
. ./cmd.sh

#set -e -u -o pipefail
. utils/parse_options.sh 

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


if [ $# -ne 0 ]; then
  exit 1
fi

dir=$exp_root/segmentation${affix}
mkdir -p $dir

# See $lang/phones.txt and decide which should be garbage
garbage_phones="<sss> <oov> <vns>"
silence_phones="SIL"
if [ ! -z $step01 ];then
for p in $garbage_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > $dir/garbage_phones.txt

for p in $silence_phones; do 
  for a in "" "_B" "_E" "_I" "_S"; do
    echo "$p$a"
  done
done > $dir/silence_phones.txt

if ! cat $dir/garbage_phones.txt $dir/silence_phones.txt | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $dir/{silence,garbage}_phones.txt"
  exit 1
fi
fi
whole_data_dir=${data_dir}_whole
whole_data_id=$(basename $whole_data_dir)

if [ ! -z $step02 ]; then
  utils/data/convert_data_dir_to_whole.sh $data_dir $whole_data_dir
fi

###############################################################################
# Extract features for the whole data directory
###############################################################################
if [ ! -z $step03 ]; then
  steps/make_mfcc.sh --nj $reco_nj --cmd "$train_cmd"  --write-utt2num-frames true \
    $whole_data_dir exp/make_mfcc/${whole_data_id}
  steps/compute_cmvn_stats.sh $whole_data_dir exp/make_mfcc/${whole_data_id}
  utils/fix_data_dir.sh $whole_data_dir
fi

###############################################################################
# Prepare SAD targets for recordings
###############################################################################
targets_dir=$dir/${whole_data_id}_combined_targets_sub3
if [ ! -z $step04 ]; then
  source-md/egs/speech_activity_detection_using_nnet3/steps/segmentation/prepare_targets_gmm.sh \
    --steps  $prepare_targets_stage \
    --train-cmd "$train_cmd" --decode-cmd "$decode_cmd" \
    --nj $nj --reco-nj $reco_nj --lang-test $lang_test \
    --garbage-phones-list $dir/garbage_phones.txt \
    --silence-phones-list $dir/silence_phones.txt \
    --merge-weights "$merge_weights" \
    --graph-dir "$graph_dir" \
    $lang $data_dir $whole_data_dir $sat_model_dir $model_dir $dir
fi

whole_data_dir=$exp_root/segmentation${affix}/train_whole
if [ ! -z $step05 ]; then
  utils/copy_data_dir.sh ${whole_data_dir} ${whole_data_dir}_hires
   steps/make_mfcc.sh --mfcc-config conf/mfcc_hires.conf --nj $reco_nj \
    ${whole_data_dir}_hires
  steps/compute_cmvn_stats.sh ${whole_data_dir}_hires
fi

if [ ! -z $step06 ]; then
  # Train a TDNN-LSTM network for SAD
  source-md/egs/speech_activity_detection_using_nnet3/train_lstm_asr_sad_1b.sh \
    --stage $nstage --train-stage $train_stage \
    --affix "1b" \
    --targets-dir $targets_dir \
    --data-dir ${whole_data_dir}_hires \
    --dir $exp_root/segmentation$affix/tdnn_lstm_asr_sad$affix 
fi

if [ ! -z $step07 ]; then
  # The options to this script must match the options used in the 
  # nnet training script. 
  # e.g. extra-left-context is 70, because the model is an LSTM trained with a 
  # chunk-left-context of 60. 
  # Note: frames-per-chunk is 150 even though the model was trained with 
  # chunk-width of 20. This is just for speed.
  # See the script for details of the options.
 for part in $test_set;do
   steps/segmentation/detect_speech_activity.sh \
    --extra-left-context 70 --extra-right-context 0 --frames-per-chunk 150 \
    --extra-left-context-initial 0 --extra-right-context-final 0 \
    --nj $test_nj --acwt 0.3 --stage $test_stage --cmd "$train_cmd" \
    data/$part \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix} \
    mfcc_hires \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/$part \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/${part}_seg

 done
fi

if [ ! -z $step08 ]; then
  # Do some diagnostics
   for part in  $test_set;do
  steps/segmentation/evaluate_segmentation.pl data/$part/segments \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/${part}_seg/segments &> \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad_1a${affix}/${part}_seg/evalutate_segmentation.log

  steps/segmentation/convert_utt2spk_and_segments_to_rttm.py \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/${part}_seg/utt2spk \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/${part}_seg/segments \
    $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/${part}_seg/sys.rttm
 done

  #export PATH=$PATH:$KALDI_ROOT/tools/sctk/bin
  #md-eval.pl -c 0.25 -r $dev10h_rttm_file \
  #  -s exp/segmentation_1a/tdnn_lstm_asr_sad_1a/dev10h_seg/sys.rttm > \
  #  exp/segmentation_1a/tdnn_lstm_asr_sad_1a/dev10h_seg/md_eval.log
fi

if [ ! -z $step09 ]; then
  for part in $test_set;do
  utils/copy_data_dir.sh $exp_root/segmentation${affix}/tdnn_lstm_asr_sad${affix}/${part}_seg \
    data/${part}_seg_asr_sad${affix}
 done

fi
  
# run-4-anydecode.sh --dir dev10h.seg_tdnn_lstm_asr_sad_1a
# %WER 40.6 | 21825 101803 | 63.6 26.3 10.1 4.1 40.6 29.8 | -0.469 | exp/chain_cleaned_pitch/tdnn_flstm_sp_bi/decode_dev10h.pem/score_11/dev10h.pem.ctm.sys
# %WER 41.1 | 21825 101803 | 63.5 26.1 10.4 4.5 41.1 31.8 | -0.523 | exp/chain_cleaned_pitch/tdnn_flstm_sp_bi/decode_dev10h.seg/score_11/dev10h.seg.ctm.sys
# %WER 40.9 | 21825 101803 | 63.5 26.1 10.4 4.4 40.9 31.4 | -0.527 | exp/chain_cleaned_pitch/tdnn_flstm_sp_bi/decode_dev10h.seg_1a_tdnn_lstm_asr_sad_1a_acwt0_3/score_11/dev10h.seg_1a_tdnn_lstm_asr_sad_1a_acwt0_3.ctm.sys
# %WER 41.0 | 21825 101803 | 63.5 26.1 10.4 4.5 41.0 31.5 | -0.522 | exp/chain_cleaned_pitch/tdnn_flstm_sp_bi/decode_dev10h.seg_asr_sad_1a/score_11/dev10h.seg_asr_sad_1a.ctm.sys
