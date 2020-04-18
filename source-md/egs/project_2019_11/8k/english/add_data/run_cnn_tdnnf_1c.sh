#!/bin/bash

# 1c is same as 1b , 1c supported  three test set
#                                  gmm=tri5
#                                  bulid_tree_ali: tri4_ali
#                                  add a parameter first_lang : data/lang 
# note:
# 1. {train_set}_sp data is used as input for steps/align_fmllr_lats.sh
# 2. data/lang is used to get lang_chain
# 3. trian_set data  (e.g: it is no speed and volume disturbed data) is used as input for steps/nnet3/chain/build_tree.sh.
# # 4. tri4_ali is used as input for  steps/nnet3/chain/build_tree.sh.
. path_v2.sh

echo 
echo "## LOG: $0 $@"
echo

set -e
. cmd.sh
cmd="slurm.pl --quiet --exclude=node01,node02,node08,node05"

nj=40

affix=1a
dnn_type=cnn_tdnnf
# configs for 'chain'
stage=0
tdnn_affix=
train_stage=-10  # original -10
get_egs_stage=-10
decode_iter=
train_set=train
tree_affix=
nnet3_affix=
num_leaves=7000
# training options
frames_per_eg=150,110,100
remove_egs=false
common_egs_dir=
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.3@0.50,0'
gmm=tri5
first_lang=data/lang
tgtdir=
test_set_dir=         # test set 40 dim mfcc feature path
test_set_ivector_dir= # test set 100 dim ivector feature path
pp=true  # true if  using pronunciation probabilities and word-dependent silence probabilities in lang else flase.

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
build_tree_ali_dir=$tgtdir/exp/tri4_ali  # used to make a new tree for chain topology, should match train data
dir=$tgtdir/exp/chain${nnet3_affix}/tdnn_${dnn_type}${tdnn_affix}_sp    # dnn output path
build_tree_train_data_dir=data/${train_set}         #  no seep and volume disturbed data ,so it is 13 mfcc feature in gmm-hmm.
lores_train_data_dir=data/${train_set}_sp           # seep disturbed data   
train_data_sp_hires_dir=data/${train_set}_sp_hires  # speed and volume disturbed data  40 dim mfcc

ivector_train_set=${train_set}_sp 
train_ivector_dir=data/nnet3_ubs2020_cs_big/ivectors_${ivector_train_set}_hires     #  train set 100 dim ivector folder

gmm_dir=$tgtdir/exp/$gmm   # used to get training lattices (for chain supervision)
treedir=$tgtdir/exp/chain${nnet3_affix}/tree_${dnn_type}${tree_affix} # new build tree output path
lat_dir=$tgtdir/exp/chain${nnet3_affix}/tri4_${train_set}_sp_lats  # training lattices directory,regenerate
                                                            # lattice path
lang=$tgtdir/data/local/lang_chain ##Specify the directory of newly generated lang, and you can modify it yourself.

if [ $stage -le 1 ]; then
  # Get the alignments as lattices (gives the LT-MMI chain training more freedom).
  # use the same num-jobs as the alignments
  nj=$(cat $build_tree_ali_dir/num_jobs) || exit 1;
  steps/align_fmllr_lats.sh --nj $nj --cmd "$cmd" $lores_train_data_dir \
    $first_lang $gmm_dir $lat_dir || exit 1;
  rm $lat_dir/fsts.*.gz # save space
fi


if [ $stage -le 2 ]; then
  # Create a version of the lang/ directory that has one state per phone in the
  # topo file. [note, it really has two states.. the first one is only repeated
  # once, the second one has zero or more repeats.]
  echo "$0: making chain lang";
  rm -rf $lang
  cp -r $first_lang $lang
  silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
  nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
  # Use our special topology... note that later on may have to tune this
  # topology.
  steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
fi

if [ $stage -le 3 ]; then
  #build_tree_train_data_dir=$train_data_dir  #  no speed and volume perturbation data ,so it is 13 mfcc feature in gmm-hmm.
  # Build a tree using our new topology.
  steps/nnet3/chain/build_tree.sh --frame-subsampling-factor 3 \
      --context-opts "--context-width=2 --central-position=1" \
      --cmd "$cmd" $num_leaves $build_tree_train_data_dir $lang $build_tree_ali_dir $treedir || exit 1;
fi

if [ $stage -le 4 ]; then
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

if [ $stage -le 5 ];then
   #--feat.online-ivector-dir ${root_dir}/exp/nnet3/ivectors_${train_set}  it is the  ivector path  of train set.
   #--feat-dir ${root_dir}/data/${train_set}_hires   This folder is a directory for storing high resolution MFCC feature data(dim=43) (volume disturbance). Corresponding to AM neural network input dim=43 ,name=input
   #--feat-dir data/${train_set}_hires_nopitch  This folder is a directory for storing high resolution MFCC feature data(dim=40) (volume disturbance).Corresponding to AM neural network input dim=40 ,name=input
   steps/nnet3/chain/train.py --stage $train_stage \
    --cmd "slurm.pl --quiet --exclude=node01,node02,node05,node04,node03" \
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
    --feat-dir $train_data_sp_hires_dir \
    --tree-dir $treedir \
    --lat-dir $lat_dir \
    --dir $dir  || exit 1;

fi

test_set_1="wiz3ktest dev_imda_part3_ivr"
test_set_2="msf_baby_bonus-8k  ubs2020_dev_cs"
test_set_3="ubs2020_dev_eng ubs2020_dev_man"
if [ $stage -le 6 ]; then
 if $pp ;then
  graph_dir=$dir/graph_better_pp
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/lang_test_pp $dir $graph_dir
  # remove <UNK> from the graph, and convert back to const-FST.
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/HCLG.fst - | \
    fstconvert --fst_type=const > $graph_dir/temp.fst
  mv $graph_dir/temp.fst $graph_dir/HCLG.fst
  rm $dir/.error 2>/dev/null || true

  for decode_set in $test_set_1 $test_set_2 $test_set_3; do
      
      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_better_pp || exit 1;
        

   done
   wait
  
   else
  graph_dir=$dir/graph_better
  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/local/lang_test $dir $graph_dir
  # remove <UNK> from the graph, and convert back to const-FST.
  fstrmsymbols --apply-to-output=true --remove-arcs=true "echo 3|" $graph_dir/HCLG.fst - | \
    fstconvert --fst_type=const > $graph_dir/temp.fst
  mv $graph_dir/temp.fst $graph_dir/HCLG.fst
  # nspk=$(wc -l <data/${decode_set}_hires/spk2utt
  rm $dir/.error 2>/dev/null || true
  for decode_set in $test_set_1 $test_set_2 $test_set_3; do
      
      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      # if decode_nj is very big(e.g: 2716) , it may decode incorrectly, you should reduce nj.
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_better || exit 1;
       
  done                                                                                                                                                       
  wait

 fi
fi

# 2020/2/21, I use mao's lm with adaption model. then make HCLG.fst
# pure english 4gram: /home4/mtz503/w2020k/4-gram-ubs2020/english-4gram/lm4-kn-adapt0.8.gz
# pure mandarin 4gram:/home4/mtz503/w2020k/4-gram-ubs2020/mandarin-4gram/lm4-kn-adapt0.8-161.gz
# the biggest 4gram:(text:cn/en/cs) /home4/mtz503/w2020k/4-gram-ubs2020/lm4-bigbig/text-1/lm4-kn-adapt-0.8_pruned.gz
dictdir=$tgtdir/data/ubs_dictv0.3
graph_dir=$dir/graph_mao_pure_english_adaption_v1_pp
if [ $stage -le 7 ]; then
   utils/format_lm.sh $tgtdir/data/lang_pp /home4/mtz503/w2020k/4-gram-ubs2020/english-4gram/lm4-kn-adapt0.8.gz  \
     $dictdir/lexiconp.txt $tgtdir/data/lang_test_pure_english_adaption_v1_pp

  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/lang_test_pure_english_adaption_v1_pp $dir $graph_dir
  rm $dir/.error 2>/dev/null || true
fi

if [ $stage -le 8 ];then
  for decode_set in ubs2020_dev_eng ; do

      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      source-md/analaysis/analysis_asr_results/decode_cer.sh  --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_pure_english_adaption_v1_pp

  done
  wait
fi

graph_dir=$dir/graph_mao_pure_mandarin_adaption_v1_pp
if [ $stage -le 9 ]; then
   utils/format_lm.sh $tgtdir/data/lang_pp /home4/mtz503/w2020k/4-gram-ubs2020/mandarin-4gram/lm4-kn-adapt0.8-161.gz  \
     $dictdir/lexiconp.txt $tgtdir/data/lang_test_pure_mandarin_adaption_v1_pp

  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/lang_test_pure_mandarin_adaption_v1_pp $dir $graph_dir
  rm $dir/.error 2>/dev/null || true
fi

if [ $stage -le 10 ];then
  for decode_set in ubs2020_dev_man ; do

      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      source-md/analaysis/analysis_asr_results/decode_cer.sh  --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_pure_mandarin_adaption_v1_pp

  done
  wait
fi

graph_dir=$dir/graph_mao_cs_adaption_v1_pp
if [ $stage -le 11 ]; then
   utils/format_lm.sh $tgtdir/data/lang_pp  /home4/mtz503/w2020k/4-gram-ubs2020/lm4-bigbig/text-1/lm4-kn-adapt-0.8_pruned.gz  \
     $dictdir/lexiconp.txt $tgtdir/data/lang_test_cs_adaption_v1_pp

  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/lang_test_cs_adaption_v1_pp $dir $graph_dir
  rm $dir/.error 2>/dev/null || true
fi

if [ $stage -le 12 ];then
  for decode_set in ubs2020_dev_man  ubs2020_dev_eng ubs2020_dev_cs; do

      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      source-md/analaysis/analysis_asr_results/decode_cer.sh  --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_cs_adaption_v1_pp

  done
  wait
fi


# test msf_baby_bonus-8k
if [ $stage -le 13 ];then
  for decode_set in msf_baby_bonus-8k; do

      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      source-md/analaysis/analysis_asr_results/decode_cer.sh  --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_cs_adaption_v1_pp

  done
  wait
fi 

# 2020-3-3
# the new lang directory that adds 1606 words into dictv0.3 :/home4/mtz503/w2020k/4-gram-ubs2020/langv0.4
# dictionary =/home4/mtz503/w2020k/4-gram-ubs2020/dictv0.4
graph_dir=$dir/graph_mao_cs_adaption_v2
if [ $stage -le 14 ]; then
   utils/format_lm.sh /home4/mtz503/w2020k/4-gram-ubs2020/langv0.4  /home4/mtz503/w2020k/4-gram-ubs2020/lm4-bigbig/text-1/lm4-kn-adapt-0.8_pruned.gz  \
     /home4/mtz503/w2020k/4-gram-ubs2020/dictv0.4/lexiconp.txt  $tgtdir/data/lang_test_cs_adaption_v2

  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/lang_test_cs_adaption_v2 $dir $graph_dir
  rm $dir/.error 2>/dev/null || true
fi


if [ $stage -le 15 ];then
  for decode_set in  ubs2020_dev_man  ubs2020_dev_eng ubs2020_dev_cs msf_baby_bonus-8k; do

      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      source-md/analaysis/analysis_asr_results/decode_cer.sh  --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_cs_adaption_v2

  done
  wait

# [md510@node08 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_eng_cs_adaption_v2/scoring_kaldi/best_cer 
# %WER 30.50 [ 8994 / 29486, 794 ins, 3008 del, 5192 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_eng_cs_adaption_v2/cer_10_0.0
#[md510@node08 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_cs_adaption_v2/scoring_kaldi/best_cer 
#%WER 22.24 [ 9057 / 40723, 1391 ins, 2682 del, 4984 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_cs_adaption_v2/cer_9_0.0
#[md510@node08 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_cs_adaption_v2/scoring_kaldi/best_cer 
#%WER 23.30 [ 14066 / 60368, 2175 ins, 4031 del, 7860 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_cs_adaption_v2/cer_9_0.0
#[md510@node08 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_cs_adaption_v2/scoring_kaldi/best_cer 
#%WER 20.82 [ 7250 / 34814, 813 ins, 1715 del, 4722 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_cs_adaption_v2/cer_8_0.0
fi


# 2020-3-3
# the new lang directory that adds 1606 words into dictv0.3 :/home4/mtz503/w2020k/4-gram-ubs2020/langv0.4
# dictionary =/home4/mtz503/w2020k/4-gram-ubs2020/dictv0.4
# LM : /home4/mtz503/w2020k/4-gram-ubs2020/lm4-bigbig/text-1/lm4-kn-adapt-0.8_new.gz
graph_dir=$dir/graph_mao_cs_adaption_v3
if [ $stage -le 16 ]; then
   utils/format_lm.sh /home4/mtz503/w2020k/4-gram-ubs2020/langv0.4  /home4/mtz503/w2020k/4-gram-ubs2020/lm4-bigbig/text-1/lm4-kn-adapt-0.8_new.gz  \
     /home4/mtz503/w2020k/4-gram-ubs2020/dictv0.4/lexiconp.txt  $tgtdir/data/lang_test_cs_adaption_v3

  # Note: it might appear that this $lang directory is mismatched, and it is as
  # far as the 'topo' is concerned, but this script doesn't read the 'topo' from
  # the lang directory.
  utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data/lang_test_cs_adaption_v3 $dir $graph_dir
  rm $dir/.error 2>/dev/null || true
fi


if [ $stage -le 17 ];then
  for decode_set in  ubs2020_dev_man  ubs2020_dev_eng ubs2020_dev_cs msf_baby_bonus-8k; do

      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      source-md/analaysis/analysis_asr_results/decode_cer.sh  --acwt 1.0 --post-decode-acwt 10.0 \
          --nj 10 --cmd "$decode_cmd" $iter_opts \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_hires \
          $graph_dir $test_set_dir/${decode_set}_hires \
          $dir/decode_${decode_set}_cs_adaption_v3

  done
  wait


# [md510@node07 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_eng_cs_adaption_v3/scoring_kaldi/best_cer 
# %WER 30.51 [ 8996 / 29486, 797 ins, 3018 del, 5181 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_eng_cs_adaption_v3/cer_10_0.0
# [md510@node07 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_cs_adaption_v3/scoring_kaldi/best_cer 
# %WER 22.27 [ 9068 / 40723, 1390 ins, 2676 del, 5002 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_cs_adaption_v3/cer_9_0.0
# [md510@node07 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_cs_adaption_v3/scoring_kaldi/best_cer 
# %WER 23.31 [ 14072 / 60368, 2176 ins, 4021 del, 7875 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_cs_adaption_v3/cer_9_0.0
# [md510@node07 add_data]$ cat run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_cs_adaption_v3/scoring_kaldi/best_cer 
# %WER 20.76 [ 7229 / 34814, 810 ins, 1711 del, 4708 sub ] run_ubs2020_cs_big/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_cs_adaption_v3/cer_8_0.0

fi
