#!/bin/bash

. ./path.sh
. ./cmd.sh



steps=

. ./utils/parse_options.sh || exit 1;

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

 # Set bash to 'debug' mode, it will exit on :
 # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
 set -e
# set -u
 set -o pipefail
data=data
[ -d $data ] || mkdir -p $data
if [ ! -z $step01 ];then
  # copy sre2004 kaldi format data
  cp -r /home3/md510/w2019a/kaldi-recipe/speaker_diarization/data/sre2004 $data/  || exit 1;
  utils/utt2spk_to_spk2utt.pl $data/sre2004/utt2spk > $data/sre2004/spk2utt 
  cat data/sre2004/spk2gender | sort | uniq > data/sre2004/spk2gender_sort
  mv data/sre2004/spk2gender_sort data/sre2004/spk2gender 
fi

# prepare 20 dimension feature 
if [ ! -z $step02 ];then
   for name in sre2004; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc_diarization_xvector.conf --nj 40 \
      --cmd "$train_cmd" --write-utt2num-frames true data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done 
fi

# prepare vad feature
if [ ! -z $step03 ];then
  for name in sre2004; do
    sid/compute_vad_decision.sh --nj 40 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done

fi

# step04,05 is configuration for diarization
# its purpose is to further cut of segment.
# I now only train gender recognition. I don't need to smaller segment.
# So I learn from sre16 recipe.
# prepare cmn feature
#if [ ! -z $step04 ];then
#   for name in sre2004; do
#    local/nnet3/xvector/prepare_feats.sh --nj 40 --cmd "$train_cmd" \
#      data/$name data/${name}_cmn exp/${name}_cmn
#    cp data/$name/vad.scp data/${name}_cmn/
#    utils/fix_data_dir.sh data/${name}_cmn
#  done
#
#  echo "0.01" > data/sre2004_cmn/frame_shift
#  # Create segments to extract x-vectors from for CNN gender training data.
#  # The segments are created using an energy-based speech activity
#  # detection (SAD) system, but this is not necessary.  You can replace
#  # this with segments computed from your favorite SAD.
#  diarization/vad_to_segments.sh --nj 40 --cmd "$train_cmd" \
#    data/sre2004_cmn data/sre2004_cmn_segmented
#
#fi
#
## get x-vectors feature for SRE2004
#if [ ! -z $step05 ];then
#  # Extract x-vectors for the SRE2004, which is our CNN gender training
#  # data.  A long period is used here so that we don't compute too
#  # many x-vectors for each recording.
#  # x-vector model
#  ellerao_nnet_dir=/home3/md510/w2019a/kaldi-recipe/speaker_diarization/exp/ellerao_retrain_xvector_model
#  nnet_dir=exp/ellerao_retrain_xvector_model
#  [ -d $nnet_dir ] || mkdir -p $nnet_dir
#  cp -r $ellerao_nnet_dir/*.raw  $nnet_dir
#  cp -r $ellerao_nnet_dir/configs $nnet_dir
#  cp -r $ellerao_nnet_dir/extract.config $nnet_dir
#  cp -r $ellerao_nnet_dir/nnet.config $nnet_dir
#  cp -r $ellerao_nnet_dir/max_chunk_size $nnet_dir
#  cp -r $ellerao_nnet_dir/min_chunk_size $nnet_dir
#
#  diarization/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 10G" \
#    --nj 40 --window 3.0 --period 10.0 --min-segment 1.5 --apply-cmn false \
#    --hard-min true $nnet_dir \
#    data/sre2004_cmn_segmented $nnet_dir/xvectors_sre2004_segmented
#
#fi

if [ ! -z $step06 ];then
   # x-vector model
  ellerao_nnet_dir=/home3/md510/w2019a/kaldi-recipe/speaker_diarization/exp/ellerao_retrain_xvector_model
  nnet_dir=exp/ellerao_retrain_xvector_model
  [ -d $nnet_dir ] || mkdir -p $nnet_dir
  cp -r $ellerao_nnet_dir/*.raw  $nnet_dir
  cp -r $ellerao_nnet_dir/configs $nnet_dir
  cp -r $ellerao_nnet_dir/extract.config $nnet_dir
  cp -r $ellerao_nnet_dir/nnet.config $nnet_dir
  cp -r $ellerao_nnet_dir/max_chunk_size $nnet_dir
  cp -r $ellerao_nnet_dir/min_chunk_size $nnet_dir

   sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd --mem 6G" --nj 40 \
    $nnet_dir data/sre2004 \
    exp/xvectors_sre2004

fi
