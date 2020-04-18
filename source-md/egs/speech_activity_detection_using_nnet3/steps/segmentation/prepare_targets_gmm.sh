#! /bin/bash

# Copyright 2017  Vimal Manohar
# Apache 2.0
  
# This script prepares targets for training neural network for 
# speech activity detction. 
# See steps/segmentation/lats_to_targets.sh for details about the 
# format of the targets.

# The targets are obtained from a combination
# of supervision-constrained lattices and lattices obtained by decoding. 
# Also, we assume that the out-of-segment regions are all silence (target 
# values of [ 1 0 0 ]. We merge the targets from the multiple sources 
# by a weighted average using weights specified by --weights. Also, 
# the frames where the labels from multiple sources do not match are 
# removed in the script steps/segmentation/merge_targets_dirs.sh.

# In this script, we use GMMs trained for ASR on in-domain data 
# to generate the lattices required for creating the targets. To generate
# supervision-constrained lattices, we use speaker-adapted GMM models. To 
# generate lattices without supervision, we use speaker-independent GMM models
# from the LDA+MLLT stage, but apply per-recording cepstral mean subtraction.
# The phones in the lattices are mapped deterministically to 
# 0, 1, and 2 representing respectively silence, speech and garbage classes.
# The mapping is defined by --garbage-phones-list and --silence-phones-list
# options. But when these are unspecified, the silence phones other than
# oov are mapped to silence class and the oov is mapped to garbage class.

#stage=-1
steps=1
train_cmd=run.pl
decode_cmd=run.pl
nj=4
reco_nj=4

lang_test=    # If different from $lang
graph_dir=    # If not provided, a new one will be created using $lang_test

garbage_phones_list=
silence_phones_list=

# Uniform segmentation options for decoding whole recordings. All values are in
# seconds.
max_segment_duration=10
overlap_duration=2.5
max_remaining_duration=5  # If the last remaining piece when splitting uniformly
                          # is smaller than this duration, then the last piece 
                          # is  merged with the previous.

# List of weights on labels obtained from alignment, 
# labels obtained from decoding and default labels in out-of-segment regions
merge_weights=1.0,0.1,0.5

[ -f ./path.sh ] && . ./path.sh 

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


if [ $# -ne 6 ]; then
  cat <<EOF
  This script prepares targets for training neural network for 
  speech activity detction. The targets are obtained from a combination
  of supervision-constrained lattices and lattices obtained by decoding. 
  See comments in the script for more details.

  Usage: $0 <lang> <data> <whole-recording-data> <ali-model-dir> <model-dir> <dir>
   e.g.: $0 data/lang data/train data/train_whole exp/tri5 exp/tri4 exp/segmentation_1a
  
  Note: <whole-recording-data> is expected to have feats.scp and <data> 
  expected to have segments file. We will get the features for <data> by 
  using row ranges of <whole-recording-data>/feats.scp. This script will 
  work on a copy of <data> created to have the recording-id as the speaker-id.
EOF
  exit 1
fi

lang=$1   # Must match the one used to train the models
in_data_dir=$2
in_whole_data_dir=$3
ali_model_dir=$4  # Model directory used to align the $data_dir to get target 
                  # labels for training SAD. This should typically be a
                  # speaker-adapted system.
model_dir=$5      # Model direcotry used to decode the whole-recording version
                  # of the $data_dir to get target labels for training SAD. This
                  # should typically be a speaker-independent system like
                  # LDA+MLLT system.
dir=$6

mkdir -p $dir

if [ -z "$lang_test" ]; then
  lang_test=$lang
fi

extra_files=
if [ -z "$graph_dir" ]; then
  extra_files="$extra_files $lang_test/G.fst $lang_test/phones.txt"
else
  extra_files="$extra_files $graph_dir/HCLG.fst $graph_dir/phones.txt"
fi

if [ ! -z $step01 ];then
 echo "# now check required files"
 for f in $in_whole_data_dir/feats.scp $in_data_dir/segments \
  $lang/phones.txt $garbage_phones_list $silence_phones_list \
  $ali_model_dir/final.mdl $model_dir/final.mdl $extra_files; do
  if [ ! -f $f ]; then
    echo "$0: Could not find file $f"
    exit 1
  fi
 done
fi

if [ ! -z $step02 ];then
 echo "# validate input data, $in_data_dir and $in_whole_data_dir "
 utils/validate_data_dir.sh $in_data_dir || exit 1
 utils/validate_data_dir.sh --no-text $in_whole_data_dir || exit 1
fi
if [ ! -z $step03 ];then
 echo "# validate $garbage_phones_list  and $silence_phones_list "
 if ! cat $garbage_phones_list $silence_phones_list | \
  steps/segmentation/internal/verify_phones_list.py $lang/phones.txt; then
  echo "$0: Invalid $garbage_phones_list $silence_phones_list"
  exit 1
 fi
fi

data_id=$(basename $in_data_dir)
whole_data_id=$(basename $in_whole_data_dir)

if [ ! -z $step04 ]; then
  echo "# now modify speaker info, it is stored at $dir/$data_id "
  rm -r $dir/$data_id 2>/dev/null || true
  mkdir -p $dir/$data_id

  utils/data/modify_speaker_info_to_recording.sh \
    $in_data_dir $dir/$data_id || exit 1
  utils/validate_data_dir.sh --no-feats $dir/$data_id || exit 1
fi 

# Work with a temporary data directory with recording-id as the speaker labels.
data_dir=$dir/${data_id}

###############################################################################
# Get feats for the manual segments
###############################################################################
if [ ! -z $step05 ]; then
  echo "# step05 now Get feats for the manual segments, it is stored at $data_dir "
  utils/data/subsegment_data_dir.sh $in_whole_data_dir ${data_dir}/segments ${data_dir}/tmp
  cp $data_dir/tmp/feats.scp $data_dir

  steps/compute_cmvn_stats.sh $data_dir || exit 1
fi

if [ ! -z $step06 ]; then
  echo "# step06 now get cmvn feat, it is stored at $dir/$whole_data_id"
  utils/copy_data_dir.sh $in_whole_data_dir $dir/$whole_data_id

  utils/fix_data_dir.sh $dir/$whole_data_id

  # Copy the CMVN stats to the whole directory
  cp $data_dir/cmvn.scp $dir/$whole_data_id
fi

# Work with a temporary data directory with CMVN stats computed using 
# only the segments from the original data directory.
whole_data_dir=$dir/$whole_data_id

###############################################################################
# Obtain supervision-constrained lattices
###############################################################################
sup_lats_dir=$dir/`basename ${ali_model_dir}`_sup_lats_${data_id}

if [ ! -z $step07 ]; then
  echo "# step07 now Obtain supervision-constrained lattices, it is stored at  $sup_lats_dir "
  steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" \
    ${data_dir} ${lang} ${ali_model_dir} $sup_lats_dir || exit 1
fi

###############################################################################
# Uniformly segment whole data directory for decoding
###############################################################################
uniform_seg_data_dir=$dir/${whole_data_id}_uniformseg_${max_segment_duration}sec
uniform_seg_data_id=`basename $uniform_seg_data_dir`

if [ ! -z $step08 ]; then
  echo "# step08 now Uniformly segment whole data directory for decoding, it is stored at $uniform_seg_data_dir"
  utils/data/get_segments_for_data.sh ${whole_data_dir} > \
    ${whole_data_dir}/segments

  mkdir -p $uniform_seg_data_dir

  utils/data/get_uniform_subsegments.py \
    --max-segment-duration $max_segment_duration \
    --overlap-duration $overlap_duration \
    --max-remaining-duration $max_remaining_duration \
    ${whole_data_dir}/segments > $uniform_seg_data_dir/sub_segments

  utils/data/subsegment_data_dir.sh $whole_data_dir \
    $uniform_seg_data_dir/sub_segments $uniform_seg_data_dir
  cp $whole_data_dir/cmvn.scp $uniform_seg_data_dir/
fi

model_id=$(basename $model_dir)
###############################################################################
# Create graph dir for decoding
###############################################################################
if [ -z "$graph_dir" ]; then
  graph_dir=$dir/$model_id/graph
  if [ ! -z $step09 ]; then
    echo "# step09 now make HCLG.fst ,it is stored at $graph_dir, $model_dir is used here "
    if [ ! -f $graph_dir/HCLG.fst ]; then
      rm -r $dir/lang_test 2>/dev/null || true
      cp -r $lang_test/ $dir/lang_test
      utils/mkgraph.sh $dir/lang_test $model_dir $graph_dir || exit 1
    fi
  fi
fi

###############################################################################
# Decode uniformly segmented data directory
###############################################################################
model_id=$(basename $model_dir)
decode_dir=$dir/${model_id}/decode_${uniform_seg_data_id}
if [ ! -z $step10 ]; then 
  mkdir -p $decode_dir
  echo "# step10 now Decode uniformly segmented data directory, decoder_dir is $decode_dir"
  cp $model_dir/{final.mdl,final.mat,*_opts,tree} $dir/${model_id}
  cp $model_dir/phones.txt $dir/$model_id

  # We use a small beam and max-active since we are only interested in 
  # the speech / silence decisions, not the exact word sequences.
  steps/decode.sh --cmd "$decode_cmd --mem 2G" --nj $nj \
    --max-active 1000 --beam 10.0 \
    --decode-extra-opts "--word-determinize=false" --skip-scoring true \
    $graph_dir $uniform_seg_data_dir $decode_dir
fi

ali_model_id=`basename $ali_model_dir`
###############################################################################
# Get frame-level targets from lattices for nnet training
# Targets are matrices of 3 columns -- silence, speech and garbage
# The target values are obtained by summing up posterior probabilites of 
# arcs from lattice-arc-post over silence, speech and garbage phones.
###############################################################################
if [ ! -z $step11 ]; then
  echo "# step11  Get frame-level targets from supervision-constrained lattice for nnet training"
  echo "#         Targets are matrices of 3 columns -- silence, speech and garbage"
  echo "#         The target values are obtained by summing up posterior probabilites of "
  echo "#         arcs from lattice-arc-post over silence, speech and garbage phones."
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-phones "$silence_phones_list" \
    --garbage-phones "$garbage_phones_list" \
    --max-phone-duration 0.5 \
    $data_dir $lang $sup_lats_dir \
    $dir/${ali_model_id}_${data_id}_sup_targets
fi

if [ ! -z $step12 ]; then
  echo "# # step12 "
  steps/segmentation/lats_to_targets.sh --cmd "$train_cmd" \
    --silence-phones "$silence_phones_list" \
    --garbage-phones "$garbage_phones_list" \
    --max-phone-duration 0.5 \
    $uniform_seg_data_dir $lang $decode_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets
fi

###############################################################################
# Convert targets to be w.r.t. whole data directory and subsample the 
# targets by a factor of 3.
# Since the targets from transcript-constrained lattices have only values 
# for the manual segments, these are converted to whole recording-levels 
# by inserting [ 0 0 0 ] for the out-of-manual segment regions.
###############################################################################
if [ ! -z $step13 ]; then
  echo "# step13 Convert targets to be w.r.t. whole data directory and subsample the "
  echo "# targets by a factor of 3."
  echo "# Since the targets from transcript-constrained lattices have only values "
  echo "# for the manual segments, these are converted to whole recording-levels "
  echo "# by inserting [ 0 0 0 ] for the out-of-manual segment regions."
  steps/segmentation/convert_targets_dir_to_whole_recording.sh --cmd "$train_cmd" --nj $reco_nj \
    $data_dir $whole_data_dir \
    $dir/${ali_model_id}_${data_id}_sup_targets \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets
  
  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj $reco_nj 3 \
    $whole_data_dir \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets_sub3
fi

###############################################################################
# Convert the targets from decoding to whole recording. 
###############################################################################
if [ ! -z $step14 ]; then
  echo "# step14 Convert the targets from decoding to whole recording. "
  echo "#        its output is $dir/${model_id}_${whole_data_id}_targets and  $dir/${model_id}_${whole_data_id}_targets_sub3"
  steps/segmentation/convert_targets_dir_to_whole_recording.sh --cmd "$train_cmd" --nj $reco_nj \
    $dir/${uniform_seg_data_id} $whole_data_dir \
    $dir/${model_id}_${uniform_seg_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets

  steps/segmentation/resample_targets_dir.sh --cmd "$train_cmd" --nj $reco_nj 3 \
    $whole_data_dir \
    $dir/${model_id}_${whole_data_id}_targets \
    $dir/${model_id}_${whole_data_id}_targets_sub3
fi

###############################################################################
# "default targets" values for the out-of-manual-segment regions.
# We assume in this setup that this is silence i.e. [ 1 0 0 ].
###############################################################################

if [ ! -z $step15 ]; then
  echo "# step15 "default targets" values for the out-of-manual-segment regions."
  echo "#         We assume in this setup that this is silence i.e. [ 1 0 0 ]."
  echo " [ 1 0 0 ]" > $dir/default_targets.vec
  steps/segmentation/get_targets_for_out_of_segments.sh --cmd "$train_cmd" \
    --nj $reco_nj --frame-subsampling-factor 3 \
    --default-targets $dir/default_targets.vec \
    $data_dir $whole_data_dir $dir/out_of_seg_${whole_data_id}_default_targets_sub3
fi

###############################################################################
# Merge targets for the same data from multiple sources (systems)
# --weights is used to weight targets from alignment with a higher weight 
# the targets from decoding. 
# If --remove-mismatch-frames is true, then if alignment and decoding 
# disagree (more than 0.5 probability on different classes), then those frames
# are removed by setting targets to [ 0 0 0 ]. 
###############################################################################
if [ ! -z $step16 ]; then
  echo "# step16 now  Merge targets for the same data from multiple sources (systems)"
  echo "         # --weights is used to weight targets from alignment with a higher weight "
  echo "         # the targets from decoding. "
  echo "         # If --remove-mismatch-frames is true, then if alignment and decoding "
  echo "         # disagree (more than 0.5 probability on different classes), then those frames"
  echo "         # are removed by setting targets to [ 0 0 0 ]. "
  echo "         # its output at  $dir/${whole_data_id}_combined_targets_sub3"
  echo "         # copy $dir/${whole_data_id}_combined_targets_sub3/targets.scp to $dir"
  
  steps/segmentation/merge_targets_dirs.sh --cmd "$train_cmd" --nj $reco_nj \
    --weights $merge_weights --remove-mismatch-frames true \
    $whole_data_dir \
    $dir/${ali_model_id}_${whole_data_id}_sup_targets_sub3 \
    $dir/${model_id}_${whole_data_id}_targets_sub3 \
    $dir/out_of_seg_${whole_data_id}_default_targets_sub3 \
    $dir/${whole_data_id}_combined_targets_sub3
fi

cp $dir/${whole_data_id}_combined_targets_sub3/targets.scp $dir/

echo "$0: Prepared targets in $dir/targets.scp"
