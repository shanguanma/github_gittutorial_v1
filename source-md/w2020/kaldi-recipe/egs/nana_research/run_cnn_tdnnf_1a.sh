#!/bin/bash

#This script is modified to swbd/s5c/local/chain/tuning/run_cnn_tdnn_1a.sh

# This is based on tdnn_7q, but adding cnn as the front-end.
# The cnn-tdnn-f (cnn_tdnn_1a) outperforms the tdnn-f (tdnn_7q).
#example:
##sbatch --nodelist=node03 -o ./update-tdnn-ivector-oct23-2018/exp/run_cnn_tdnn_1a-step1.log local-source/egs/seame/chain/run_cnn_tdnn_1a.sh  --cmd "slurm.pl --quiet --exclude=node04,node05,node06"  --steps 1  update-tdnn-ivector-oct23-2018 
#   system    cnn_tdnnf_1a         cnn_tdnnf_1a+rnnlm(1b)
#   sge         25.89%                  24.68%
#   man         18.88%                  17.84%


#. cmd.sh 
. path.sh

echo 
echo "## LOG: $0 $@"
echo

set -e

cmd="slurm.pl --quiet --exclude=node01,node02,node08,node06"

nj=50

affix=1a
dnn_type=cnn_tdnnf
# configs for 'chain'
stage=0
tdnn_affix=
train_stage=-10  # original -10
get_egs_stage=-10
decode_iter=
train_set=train_aug
tree_affix=
nnet3_affix=
num_leaves=7000
# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.3@0.50,0'
gmm=tri4a
build_tree_ali_dir=exp/tri3a_ali  # used to make a new tree for chain topology, should match train data
## End configuration section.
. parse_options.sh || exit 1  #it must be stay here.

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

if ! cuda-compiled; then
  cat <<EOF && exit 1
This script is intended to be used with GPUs but you have not compiled Kaldi with CUDA
If you want to use GPUs (and have them), go to src/, and configure and make on a machine
where "nvcc" is installed.
EOF
fi




dir=exp/chain${nnet3_affix}/tdnn_${dnn_type}${tdnn_affix}_sp    # dnn output path
train_data_dir=data/${train_set}_sp_hires           # seep and volume disturbed data 
lores_train_data_dir=data/${train_set}_sp           # seep disturbed data   
build_tree_train_data_dir=data/${train_set}         #  no seep and volume disturbed data
train_ivector_dir=exp/nnet3${nnet3_affix}/ivectors_${train_set}_sp_hires
gmm_dir=exp/$gmm   # used to get training lattices (for chain supervision)
treedir=exp/chain${nnet3_affix}/tree_${dnn_type}${tree_affix} # new build tree output path
lat_dir=exp/chain${nnet3_affix}/tri4a_${train_set}_sp_lats  # training lattices directory,regenerate
                                                            # lattice path
#test_dir="dev_man dev_sge"
lang=data/lang_chain ##Specify the directory of newly generated lang, and you can modify it yourself.

# The iVector-extraction and feature-dumping parts are the same as the standard
# nnet3 setup, and you can skip them by setting "--stage 8" if you have already
# run those things.
source-md/w2020/kaldi-recipe/egs/nana_research/run_ivector_common.sh \
                                   --stage $stage  \
                                   --speed-perturb true  \
                                   --train-set $train_set \
                                   --cmd "$cmd" \
                                   --nnet3-affix $nnet3_affix || exit 1;

if [ $stage -le 9 ]; then
  # Get the alignments as lattices (gives the chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$cmd" $lores_train_data_dir \
    data/lang $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 10 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  rm -rf $lang
  cp -r data/lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 11 ]; then
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$cmd" $num_leaves $build_tree_train_data_dir $lang $build_tree_ali_dir $treedir || exit 1;
fi

if [ $stage -le 12 ]; then
  echo "$0: creating neural net configs using the xconfig parser";

  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

  cnn_opts="l2-regularize=0.01"
  ivector_affine_opts="l2-regularize=0.01"
  tdnnf_first_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.0"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.002"
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input
  # this takes the MFCCs and generates filterbank coefficients.  The MFCCs
  # are more compressible so we prefer to dump the MFCCs to disk rather
  # than filterbanks.
  # dim=40,this is output dimension in idct-layer . its details is found in steps/libs/nnet3/xconfig/basic_layers.py,Xconfig  #  IdctLayer
  # dim=200 ,this is output dimension in linear-component.
 
  idct-layer name=idct input=input dim=40 cepstral-lifter=22 affine-transform-file=$dir/configs/idct.mat

  linear-component name=ivector-linear $ivector_affine_opts dim=200 input=ReplaceIndex(ivector, t, 0)
  batchnorm-component name=ivector-batchnorm target-rms=0.025

  batchnorm-component name=idct-batchnorm input=idct
  combine-feature-maps-layer name=combine_inputs input=Append(idct-batchnorm, ivector-batchnorm) num-filters1=1 num-filters2=5 height=40

  conv-relu-batchnorm-layer name=cnn1 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64 
  conv-relu-batchnorm-layer name=cnn2 $cnn_opts height-in=40 height-out=40 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=64
  conv-relu-batchnorm-layer name=cnn3 $cnn_opts height-in=40 height-out=20 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn4 $cnn_opts height-in=20 height-out=20 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=128
  conv-relu-batchnorm-layer name=cnn5 $cnn_opts height-in=20 height-out=10 height-subsample-out=2 time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  conv-relu-batchnorm-layer name=cnn6 $cnn_opts height-in=10 height-out=10  time-offsets=-1,0,1 height-offsets=-1,0,1 num-filters-out=256
  # the first TDNN-F layer has no bypass (since dims don't match), and a larger bottleneck so the
  # information bottleneck doesn't become a problem.  (we use time-stride=0 so no splicing, to
  # limit the num-parameters).
  tdnnf-layer name=tdnnf7 $tdnnf_first_opts dim=1536 bottleneck-dim=256 time-stride=0
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf14 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  tdnnf-layer name=tdnnf15 $tdnnf_opts dim=1536 bottleneck-dim=160 time-stride=3
  linear-component name=prefinal-l dim=256 $linear_opts
  ## adding the layers for chain branch
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts
  # adding the layers for xent branch
  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts small-dim=256 big-dim=1536
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 13 ];then
   #--feat.online-ivector-dir ${root_dir}/exp/nnet3/ivectors_${train_set}  it is the  ivector path  of train set.
   #--feat-dir ${root_dir}/data/${train_set}_hires   This folder is a directory for storing high resolution MFCC feature data(dim=43) (volume disturbance). Corresponding to AM neural network input dim=43 ,name=input
   #--feat-dir data/${train_set}_hires_nopitch  This folder is a directory for storing high resolution MFCC feature data(dim=40) (volume disturbance).Corresponding to AM neural network input dim=40 ,name=input
   steps/nnet3/chain/train.py \
    --stage $train_stage \
    --cmd "slurm.pl --quiet --exclude=node01,node02,node06,node08" \
    --feat.online-ivector-dir $train_ivector_dir \
    --feat.cmvn-opts "--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient 0.1 \
    --chain.l2-regularize 0.0 \
    --chain.apply-deriv-weights false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.add-option="--optimization.memory-compression-level=2" \
    --egs.dir "$common_egs_dir" \
    --egs.stage $get_egs_stage \
    --egs.opts "--frames-overlap-per-eg 0 --constrained false" \
    --egs.chunk-width $frames_per_eg \
    --trainer.num-chunk-per-minibatch 64 \
    --trainer.frames-per-iter 1500000 \
    --trainer.num-epochs 6 \
    --trainer.optimization.num-jobs-initial 3 \
    --trainer.optimization.num-jobs-final 16 \
    --trainer.optimization.initial-effective-lrate 0.00025 \
    --trainer.optimization.final-effective-lrate 0.000025 \
    --trainer.max-param-change 2.0 \
    --cleanup.remove-egs $remove_egs \
    --feat-dir $train_data_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

