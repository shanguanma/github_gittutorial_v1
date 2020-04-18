#!/bin/bash

# Copyright 2018-2020 (Authors: zeng zhiping zengzp0912@gmail.com) 2020-03-18 updated
#
## re-edit from wsj run_tdnn_1g.sh

# . path.sh
# . cmd.sh
echo
echo "## LOG: $0 $@"
echo
# begin sub
cmd='slurm.pl --quiet'
nj=20
steps=
numleaves=9000
hidden_dim=1024
get_egs_stage=-10  #1.working out 2.get subset 4.generate 5. recombe 6.remove
num_epochs=10
chainname=chain1024tdnnf
num_jobs_initial=3
num_jobs_final=16

egs_opts=" --frames-overlap-per-eg 0 --generate-egs-scp true "
max_param_change=2.0
initial_effective_lrate=0.0005
final_effective_lrate=0.00005
leftmost_questions_truncate=-1
common_egs_dir=
train_ivectors=
bottleneck_dim=128
small_dim=192


# LSTM/chain options
train_stage=-10 #  ## (default -10) -4 egs
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=false

# end sub

. parse_options.sh || exit 1

function Example {
 cat<<EOF
 train tdnn , if there something wrong change steps(default 1-7) train_stage(default -10) to skip succ step
 steps(default 1-7) 3.topo 4.tree 5.configs 6.train
 $0 --steps 1 --cmd "$cmd" --nj $nj \
 --numleaves 9000 \
 --hidden-dim 1024\
 --chainname chain1024tdnnf \
  --train-ivectors ../acumen/update-sept-26-2017/exp/tdnn/ivector-train \
 ../acumen/update-sept-23-2017/data/semi_ntu/mfcc-pitch \
 ../acumen/update-sept-23-2017/data/semi_ntu/mfcc-hires \
 ../acumen/update-sept-26-2017/data/lang-silprob \
 ../acumen/update-sept-26-2017/exp/tri4a/ali_train \
 ../acumen/update-sept-26-2017/exp/tri4a/ali_train-lattice \
 ../acumen/update-sept-26-2017/exp/tdnn

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
if [ $# -ne 6 ]; then
  Example && exit 1
fi
train_mfcc_pitch=$1
train_mfcc_hires=$2
lang=$3
alidir=$4
alilatdir=$5
tgtdir=$6

lang_chain=$tgtdir/lang-chain
if [ ! -z $step03 ]; then
  rm -rf $lang_chain
  cp -r $lang $lang_chain
  silphonelist=$(cat $lang_chain/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang_chain/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang_chain/topo || exit 1
fi
treedir=$tgtdir/chain-tree
if [ ! -z $step04 ]; then
  # Build a tree using our new topology. This is the critically different
  # step compared with other recipes.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
  --leftmost-questions-truncate $leftmost_questions_truncate \
  --context-opts "--context-width=2 --central-position=1" \
  --cmd "$cmd" $numleaves $train_mfcc_pitch $lang_chain $alidir $treedir || exit 1
  echo "## LOG (step04): done with tree building '$treedir' @ `date`"
fi

chaindir=$tgtdir/$chainname
if [ ! -z $step05 ]; then
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print (0.5/$xent_regularize)" | python)

  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $chaindir/configs
  cat <<EOF > $chaindir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$chaindir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=$hidden_dim
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=$hidden_dim bottleneck-dim=$bottleneck_dim time-stride=3
  linear-component name=prefinal-l dim=$small_dim $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=$hidden_dim small-dim=$small_dim
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=$hidden_dim small-dim=$small_dim
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $chaindir/configs/network.xconfig --config-dir $chaindir/configs/
  echo "## LOG (step05): done with xconfig-file generation ('$chaindir/configs') @ `date`"
fi

if [ ! -z $step06 ]; then
  echo "## LOG (step06): training started @ `date`"
  steps/nnet3/chain/train.py --stage $train_stage \
    --cmd="$cmd" \
    --feat.online-ivector-dir=$train_ivectors \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=$max_param_change \
    --trainer.num-epochs=$num_epochs \
    --trainer.frames-per-iter=5000000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_initial \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=$initial_effective_lrate \
    --trainer.optimization.final-effective-lrate=$final_effective_lrate \
    --trainer.num-chunk-per-minibatch=160,128 \
    --trainer.optimization.momentum=0.0 \
    --trainer.add-option='--cuda-memory-proportion=0.7' \
    --egs.stage=$get_egs_stage \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=$chunk_left_context \
    --egs.chunk-right-context=$chunk_right_context \
    --egs.dir="$common_egs_dir" \
    --egs.opts="$egs_opts" \
    --cleanup.remove-egs=$remove_egs \
    --use-gpu=wait \
    --feat-dir=$train_mfcc_hires \
    --tree-dir=$treedir \
    --lat-dir=$alilatdir \
    --dir=$chaindir  || exit 1;
#    --chain.frame-subsampling-factor 1 \
#    --chain.alignment-subsampling-factor 1 \
#    --trainer.optimization.proportional-shrink $proportional_shrink \
  echo "## LOG (step06): tdnn done (chaindir=$chaindir) @ `date`"
fi

echo "## LOG ($0): done !"
