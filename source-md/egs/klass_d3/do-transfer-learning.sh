#!/bin/bash 

# By Haihua Xu, TL@NTU, 2019
# By Zhiping Zeng, TL@NTU, 2019
# refer to egs/rm/s5/local/chain/run_tdnn_wsj_rm.sh

echo
echo "LOG ($0): $@"
echo

set -e

. path.sh || exit 1
# begin option
cmd="slurm.pl --quiet"
xent_regularize=0.1
nj=20
steps=
train_stage=-10
egs_stage=-10
common_egs_dir=
primary_lr_factor=0.25 # learning-rate factor for all except last layer in transferred source model
phone_lm_scales="1,10"
# training options
num_epochs=8
num_jobs_initial=2
num_jobs_final=6
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0
srand=0
remove_egs=true
max_param_change=2.0
dropout_schedule='0,0@0.20,0.5@0.50,0'
# end option

. parse_options.sh || exit 1 

function Example {
 cat<<EOF
  
  Usage: $0 <src_mdl> <ivec_extractor_dir> <src_lang>

EOF
}

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


# if [ $# -ne 4 ]; then
#  Example && exit 1
# fi
train_mfcc_pitch=exp/may_09_klass_d3/data/tune_train/mfcc-pitch
train_mfcc_hires=exp/may_09_klass_d3/data/tune_train/mfcc-hires
src_lang=exp/may_09_klass_d3/data/lang_prob_pron
sdir=exp/may_09_klass_d3/exp/tri3a_tune
src_mdl=./exp/may_09_klass_d3/exp/tdnnf/final.mdl
ivector_extractor_dir=./exp/may_09_klass_d3/exp/ivector-extractor
alidir=exp/may_09_klass_d3/exp/tri3a_tune/ali_train
expdir=exp/may_09_klass_d3/exp/transfer-learning-3ksenone
[ -d expdir ] || mkdir -p $expdir

common_egs_dir=

ivector_dim=$(nnet3-am-info --print-args=false $src_mdl | grep "ivector-dim" | cut -d" " -f2)
echo "ivector_dim=$ivector_dim"


ivector_train=$expdir/ivector-train
data=$train_mfcc_hires
if [ ! -z $step01 ]; then
   utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
       $data ${data}-max2

   steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
       ${data}-max2 \
       $ivector_extractor_dir $ivector_train  
fi

# get the alignments as lattices
latdir=$sdir/ali_train_lats
if [ ! -z $step02 ]; then
 nj=$(cat $alidir/num_jobs)
 steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj \
 $train_mfcc_pitch $src_lang $sdir    $latdir || exit 1
 rm $latdir/fsts.*.gz 2>/dev/null 
fi

# prepare chain lang
langdir=$expdir/lang
if [ ! -z $step03 ]; then
    echo "## LOG (step03): make chain lang"
  cp -r $src_lang $langdir
  silphonelist=$(cat $langdir/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $langdir/phones/nonsilence.csl) || exit 1;
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$langdir/topo
fi
# train tree for the new data
treedir=$expdir/tree
if [ ! -z $step04 ]; then
    echo "## LOG ($0): build tree"
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
      --cmd "$cmd" 3000 $train_mfcc_pitch  $langdir $alidir $treedir || exit 1;
fi
# prepare new layers for the target data
dir=$expdir/tdnnf
[ -d $dir ] || mkdir -p $dir
if [ ! -z $step05 ]; then
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | /usr/bin/python2)
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF

  steps/nnet3/xconfig_to_configs.py --existing-model $src_mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/
 $cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $src_mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
  
fi
# train the models
if [ ! -z $step06 ]; then

    steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
    --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir=$ivector_train \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.00005 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=6 \
    --trainer.frames-per-iter=5000000 \
    --trainer.optimization.num-jobs-initial=3 \
    --trainer.optimization.num-jobs-final=3 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
    --egs.stage=$egs_stage \
    --egs.dir="$common_egs_dir" \
    --egs.opts="--frames-overlap-per-eg 0" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=true \
    --reporting.email="$reporting_email" \
    --feat-dir=$train_mfcc_hires \
    --tree-dir=$treedir \
    --lat-dir=$latdir \
    --dir=$dir  || exit 1;
fi
