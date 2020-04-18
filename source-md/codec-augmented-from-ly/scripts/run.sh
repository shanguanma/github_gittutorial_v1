#!/bin/bash 

. path.sh 

echo
echo "## LOG ($0): $@"
echo

# begin option
cmd="slurm.pl --quiet"
steps=
nj=120

train_stage=-10
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'

# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=true

# end option

function UsageExample {
 cat <<EOF

 $0 --steps 1 --cmd "$cmd" --nj $nj \
 /home4/md510/w2018/seame/data/local/dict/lexicon.txt \
 ngaho-sg-english-i2r/data/local/dict/lexicon.txt seame2019 

EOF
}

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
if [ $# -ne 3 ]; then
  UsageExample && exit 1
fi
seame_lexicon=$1
sge_lexicon=$2
tgtdir=$3

[ -d $tgtdir ] || mkdir -p $tgtdir
dictdir=$tgtdir/data/local/dict
[ -d $dictdir ] || mkdir -p $dictdir
if [ ! -z $step01 ]; then
  grep -v '<' $seame_lexicon | \
  perl -ane 'use utf8; use open qw(:std :utf8); 
    chomp; @A = split(/\s+/); $word = shift @A;
    for($i = 0; $i < scalar @A; $i++) { 
      if($A[$i] =~ /_sge/) {
        $A[$i] =~ m/^(.*)(_sge)$/ or die;
        $phone = uc($1);
        $A[$i] = $phone . "_eng";
      }
    }
    $pron = join(" ", @A);
    print "$word\t$pron\n"; ' | \
   perl -e ' use utf8; use open qw(:std :utf8); 
     $sge_lexicon = shift @ARGV;   open(F, "$sge_lexicon") or die;
     while(<F>) { chomp; @A = split(/\s+/); $word = shift @A;
       $pron = join(" ", @A);  
       print "$word\t$pron\n";
     } close F;
     while(<STDIN>) {print;}
   ' $sge_lexicon     | sort -u > $dictdir/lexicon.txt
fi
# prepare kaldi dict
if [ ! -z $step02 ]; then
  cat $dictdir/lexicon.txt | \
  grep -v '>' | \
  perl -ane 'use utf8; use open qw(:std :utf8);
    chomp; @A = split(/\s+/); $word = shift @A; 
    for($i = 0; $i < scalar @A; $i++) { print "$A[$i]\n";  }
  ' | sort -u > $dictdir/nonsilence_phones.txt
  cp $(dirname $seame_lexicon)/silence_phones.txt $dictdir/
  cp $(dirname $seame_lexicon)/optional_silence.txt $dictdir/
  echo -n > $dictdir/extra_questions.txt
  utils/validate_dict_dir.pl $dictdir  
fi
# prepare lang
lang=$tgtdir/data/lang_base
[ -d $lang ] || mkdir -p $lang
if [ ! -z $step03 ]; then
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang
fi
# prepare training data
source_data=/home4/md510/w2018/seame/update-tdnn-ivector-oct23-2018/data/train/mfcc-pitch
train_data=$tgtdir/data/train
[ -d $train_data ] || mkdir -p $train_data
if [ ! -z $step04 ]; then
  cat $source_data/text | \
  source-scripts/egs/acumen/show-oov-with-count.pl --from=2 $dictdir/lexicon.txt > $train_data/oov-count.txt
  if [ ! -e $train_data/oov-transfer-dict-edit.txt ]; then
    cat $train_data/oov-count.txt | \
    perl -ane 'use utf8; use open qw(:std :utf8); m/^(\S+)\s+(.*)/g or next; 
    print "$1\t$1\n";'  > $train_data/transfer-dict.txt
    cp $train_data/transfer-dict.txt $train_data/oov-transfer-dict-edit.txt  
  fi
fi
# copy train_mfcc_pitch data
train_mfcc_pitch=$tgtdir/data/train/mfcc_pitch
train_mfcc_hires=$tgtdir/data/train/mfcc_hires
[ -d $train_mfcc_pitch ] || mkdir -p $train_mfcc_pitch
[ -d $train_mfcc_hires ] || mkdir -p $train_mfcc_hires
if [ ! -z $step05 ]; then
  cp $source_data/* $train_mfcc_pitch
  cat $source_data/text | \
  source-scripts/egs/mipitalk/transfer-utterance-with-dict-ignore-oov.pl \
  $train_data/oov-transfer-dict-edit.txt > $train_mfcc_pitch/text

  cat $train_mfcc_pitch/text | \
  source-scripts/egs/acumen/show-oov-with-count.pl --from=2 \
  $dictdir/lexicon.txt > $train_mfcc_pitch/oov-count.txt

  utils/subset_data_dir.sh --utt-list $train_mfcc_pitch/text \
  /home4/md510/w2018/seame/update-tdnn-ivector-oct23-2018/data/train_sp_hires_nopitch \
  $train_mfcc_hires
  cp $train_mfcc_pitch/text $train_mfcc_hires
fi
# train gmm model
hmmdir=$tgtdir/exp/gmm-hmm
sdir=$hmmdir/tri4a
if [ ! -z $step06 ]; then
   source-scripts/egs/swahili/run-gmm-v2.sh --cmd "$cmd" --nj $nj \
  --steps 1-5 \
  --state-num 4500 \
  --pdf-num 80000 \
  $train_mfcc_pitch $lang $hmmdir  || exit 1
fi 
# train ivector extractor. This script will be depreciated
dir=$tgtdir/exp
if [ ! -z $step07 ]; then
  ./source-scripts/egs/seame-2019/run-ivector-common.sh --steps 7-10 --cmd "$cmd" --nj $nj \
  --datadir $tgtdir/data/train \
  $train_mfcc_hires $lang $sdir $dir || exit 1
fi
# perturb mfcc-pitch data and make alignment 
sp_train_mfcc_pitch=$tgtdir/data/train/mfcc_pitch_sp
if [ ! -z $step08 ]; then
  utils/data/perturb_data_dir_speed_3way.sh $train_mfcc_pitch $sp_train_mfcc_pitch
  utils/data/perturb_data_dir_volume.sh $sp_train_mfcc_pitch
fi
# extract mfcc-pitch features for perturbed data
traindata=$tgtdir/data/train
if [ ! -z $step09 ]; then
  steps/make_mfcc_pitch.sh --cmd "$cmd"  --nj $nj --mfcc-config conf/mfcc.conf \
  --pitch-config conf/pitch.conf \
  $sp_train_mfcc_pitch $traindata/feat/mfcc-pitch-sp/log \
  $traindata/feat/mfcc-pitch-sp/data  || exit 1
  steps/compute_cmvn_stats.sh $sp_train_mfcc_pitch $traindata/feat/mfcc-pitch-sp/log \
  $traindata/feat/mfcc-pitch-sp/data
  utils/fix_data_dir.sh $sp_train_mfcc_pitch 
fi
# do forced-alignment
alidir=$sdir/ali_train_sp
if [ ! -z $step10 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $sp_train_mfcc_pitch $lang  $sdir $alidir || exit 1
fi
# do lattice alignment for chain modeling 
latdir=$sdir/ali_train_lat-sp
if [ ! -z $step11 ]; then
  steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj \
  $sp_train_mfcc_pitch $lang  $sdir $latdir || exit 1
  rm $latdir/fsts.*.gz 2>/dev/null
fi
# make lang for chain models
chain_lang=$tgtdir/exp/chain_lang
if [ ! -z $step12 ]; then
  source-scripts/egs/ngaho-sg-english-i2r-2018/make_chain_lang.sh \
  $lang $chain_lang
fi
# build decision tree for chain lang 
treedir=$tgtdir/exp/chain_tree
if [ ! -z $step13 ]; then
    steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$cmd" 4500  $sp_train_mfcc_pitch  \
    $chain_lang $alidir  $treedir || exit 1
fi

# creating neural net configs using the xconfig parser
dir=$tgtdir/exp/chain-tdnn-f
[ -d $dir ] || mkdir -p $dir
if [ ! -z $step14 ]; then
  
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | /usr/bin/python2)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi
train_ivector_dir=$tgtdir/exp/ivector-train
train_mfcc_hires=$traindata/mfcc-hires-sp
# train tdnn-f models
if [ ! -z $step15 ]; then
  steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=10 \
    --trainer.frames-per-iter=5000000 \
    --trainer.optimization.num-jobs-initial=8 \
    --trainer.optimization.num-jobs-final=16 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
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
# extract ivector feature for test data
# prepare lm data
lmdir=$tgtdir/data/local/lm4
[ -d $lmdir ] || mkdir -p $lmdir
if [ ! -z $step16 ]; then
  cat $traindata/mfcc_pitch/text | \
  cut -d' ' -f2- | \
  gzip -c > $lmdir/train-text.gz
  grep -vP '#|<eps>' $lang/words.txt  | \
  awk '{print $1;}' | gzip -c > $lmdir/vocab.gz
fi
# build language models
if [ ! -z $step17 ]; then
  ngram-count -order 4 -kndiscount -interpolate\
  -vocab $lmdir/vocab.gz  -unk -sort -text $lmdir/train-text.gz -lm $lmdir/lm4.gz 
fi
# make grammar under chain_lang
chain_lang_test=${chain_lang}-test
if [ ! -z $step18 ]; then
  source-scripts/egs/acumon-d5/arpa2G.sh $lmdir/lm4.gz $chain_lang ${chain_lang}-test
fi
# make HCLG.fst
graphdir=$dir/graph
if [ ! -z $step19 ]; then
  utils/mkgraph.sh ${chain_lang}-test $dir  $graphdir
fi
# prepare dev data
if [ ! -z $step20 ]; then
  for dataname in sge man; do
      source_data=/home4/md510/w2018/seame/update-tdnn-ivector-oct23-2018/data/dev_${dataname}
      sdata=$tgtdir/data/dev-${dataname}
      data=$sdata/mfcc-hires feat=$sdata/feat/mfcc-hires/data  log=$sdata/feat/mfcc-hires/log
      [ -d $sdata ] || mkdir -p $sdata
      utils/data/copy_data_dir.sh $source_data $sdata
      [ -d $data ] || mkdir -p $data
      utils/data/copy_data_dir.sh $sdata $data
      steps/make_mfcc.sh --cmd "$cmd"  --nj 10 --mfcc-config conf/mfcc_hires.conf \
      $data  $log $feat || exit 1
      steps/compute_cmvn_stats.sh $data $log $feat || exit 1
      utils/fix_data_dir.sh $data
  done
fi
# extract ivector for two data sets

if [ ! -z $step21 ]; then
  for x in {sge,man}; do
    data=$tgtdir/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/exp/ivector-dev-${x}
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
     $data  $tgtdir/exp/ivector-extractor  $dev_ivector || exit 1
  done  
fi

# decode & evaluate
if [ ! -z $step22 ]; then
  for x in sge man; do
    data=$tgtdir/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/exp/ivector-dev-${x}
    decode_dir=$dir/decode-dev-${x}
    steps/nnet3/decode.sh --nj 10 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir $dev_ivector \
    $graphdir $data  $decode_dir  || exit 1
  done
fi

# merge data
# ngaho-sg-english-i2r/data/train/sp-mfcc-pitch
# and seame2019/data/train/mfcc_pitch_sp
# and retrain the model again to see how much benefit 
# we can get from the former for the latter in terms of 
# Singapore English WER reduction.
spm_mfcc_pitch=seame2019/data/i2r-rasp-seame/mfcc_pitch_sp
# 'spm' represents speech-perbutbed on merged data.
if [ ! -z $step40 ]; then
  utils/combine_data.sh $spm_mfcc_pitch ngaho-sg-english-i2r/data/train/sp-mfcc-pitch \
  seame2019/data/train/mfcc_pitch_sp  
fi
spm_alidir=seame2019/exp/gmm-hmm/tri4a/ali_i2r-rasp-seame-sp
if [ ! -z $step41 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $spm_mfcc_pitch seame2019/data/lang_base seame2019/exp/gmm-hmm/tri4a \
  $spm_alidir || exit 1  
fi
# train sat gmm model again
sdir=i2r-rasp-seame2019/exp/gmm-hmm/tri4a
if [ ! -z $step42 ]; then
  steps/train_sat.sh --cmd "$cmd" \
  4500 90000 $spm_mfcc_pitch seame2019/data/lang_base \
  $spm_alidir $sdir || exit 1
fi
lang=seame2019/data/lang_base
alidir=$sdir/ali_train-sp
if [ ! -z $step43 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $spm_mfcc_pitch $lang $sdir $alidir || exit 1 
fi
# ali lattice
latdir=$sdir/ali_lattice-sp
if [ ! -z $step44 ]; then
    steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj \
  $spm_mfcc_pitch $lang  $sdir $latdir || exit 1
  rm $latdir/fsts.*.gz 2>/dev/null
fi
# combine mfcc_hires features
spm_mfcc_hires=seame2019/data/i2r-rasp-seame/mfcc_hires_sp
# from data
# seame2019/data/train/mfcc-hires-sp
# ngaho-sg-english-i2r/exp/nnet3/ivector-extractor/sp-train/mfcc-hires
if [ ! -z $step50 ]; then
  utils/combine_data.sh $spm_mfcc_hires \
  seame2019/data/train/mfcc-hires-sp \
  ngaho-sg-english-i2r/exp/nnet3/ivector-extractor/sp-train/mfcc-hires
fi

# train online ivector extractor
tgtdir=i2r-rasp-seame2019/exp
if [ ! -z $step51 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  --max-utts 20000 --subsample 2 \
  $spm_mfcc_hires \
  $tgtdir/pca_transform
fi
# train the diagonal UBM
if [ ! -z $step52 ]; then
   steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj $nj \
    --num-frames 1500000 \
    --num-threads 10 \
    $spm_mfcc_hires 512 \
    $tgtdir/pca_transform $tgtdir/diag_ubm
fi
# train ivector extractor
if [ ! -z $step53 ]; then
   steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj $nj \
  --num-processes 1 \
    $spm_mfcc_hires  $tgtdir/diag_ubm  $tgtdir/ivector-extractor || exit 1;
fi

# make ivectors for the training data
ivector_train=$tgtdir/ivector-train
if [ ! -z $step54 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
  $spm_mfcc_hires ${spm_mfcc_hires}-max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
  ${spm_mfcc_hires}-max2 \
  $tgtdir/ivector-extractor $ivector_train
fi
# prepare to train chain model
# make lang for chain models
chain_lang=$tgtdir/chain_lang
if [ ! -z $step55 ]; then
  source-scripts/egs/ngaho-sg-english-i2r-2018/make_chain_lang.sh \
  $lang $chain_lang
fi
# build decision tree for chain lang 
treedir=$tgtdir/chain_tree
if [ ! -z $step56 ]; then
    steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$cmd" 4500  $spm_mfcc_pitch  \
    $chain_lang $spm_alidir  $treedir || exit 1
fi
# train chain models
dir=$tgtdir/chain-tdnn-f
[ -d $dir ] || mkdir -p $dir
if [ ! -z $step57 ]; then
    
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | /usr/bin/python2)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi
train_ivector_dir=$ivector_train
train_mfcc_hires=$spm_mfcc_hires
if [ ! -z $step58 ]; then
    steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=10 \
    --trainer.frames-per-iter=5000000 \
    --trainer.optimization.num-jobs-initial=2 \
    --trainer.optimization.num-jobs-final=8 \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
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
# make graph
chain_lang_test=${chain_lang}-test
[ -d $chain_lang_test ] || mkdir -p $chain_lang_test
graphdir=$dir/graph
if [ ! -z $step59 ]; then
  cp seame2019/exp/chain_lang-test/G.fst $chain_lang_test/
  cp -rL $chain_lang/* $chain_lang_test/
  utils/mkgraph.sh $chain_lang_test $dir  $graphdir
fi
# extract ivector for the two data sets
if [ ! -z $step60 ]; then
  for x in {sge,man}; do
    data=seame2019/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/ivector-dev-${x}
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
    $data $tgtdir/ivector-extractor $dev_ivector
  done
fi
# decode & evaluate

if [ ! -z $step61 ]; then
  for x in sge man; do
    data=seame2019/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/ivector-dev-${x}
    decode_dir=$dir/decode-dev-${x}
    steps/nnet3/decode.sh --nj 10 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir $dev_ivector \
    $graphdir $data  $decode_dir  || exit 1
  done
fi

## retrain acumon system with data augmenation and using 
## more data from seame and i2r-raspi
tgtdir=acumon-seame-i2r-raspi
datadir=$tgtdir/data
[ -d $datadir ] || mkdir -p $datadir
sp_mfcc_pitch=$datadir/acumon-train/mfcc-pitch-sp
if [ ! -z $step80 ]; then
  for x in /home3/zpz505/w2018/acumen/acumon-d5-final/data/train/mfcc-pitch; do
     sdata=$datadir/acumon-train
     utils/copy_data_dir.sh $x $sdata
     utils/data/perturb_data_dir_speed_3way.sh $sdata  $sp_mfcc_pitch
     feat=$sdata/feat/mfcc-pitch-sp/data log=$sdata/feat/mfcc-pitch-sp/log
     steps/make_mfcc_pitch.sh --cmd "$cmd"  --nj $nj --mfcc-config conf/mfcc.conf \
     --pitch-config conf/pitch.conf \
     $sp_mfcc_pitch $log $feat || exit 1
     steps/compute_cmvn_stats.sh $sp_mfcc_pitch $log $feat || exit 1
     utils/fix_data_dir.sh $sp_mfcc_pitch
  done
fi
# merge with i2r-raspi & seame data
spm_mfcc_pitch=$datadir/acumon-train/spm-mfcc-pitch
if [ ! -z $step81 ]; then
 utils/combine_data.sh $spm_mfcc_pitch   seame2019/data/i2r-rasp-seame/mfcc_pitch_sp \
 $sp_mfcc_pitch
fi

# align data 
sdir=i2r-rasp-seame2019/exp/gmm-hmm/tri4a
lang=seame2019/data/lang_base
alidir=$sdir/ali_acumon-i2r-rasp-seame
if [ ! -z $step82 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $spm_mfcc_pitch $lang $sdir $alidir || exit 1
fi
# train gmm-hmm
sdir=$tgtdir/exp/gmm-hmm/tri4a
if [ ! -z $step83 ]; then
  steps/train_sat.sh --cmd "$cmd" \
  9000 180000 $spm_mfcc_pitch $lang \
  $alidir $sdir || exit 1
fi
# make forced-alignment again with the new models
alidir=$sdir/ali_train
if [ ! -z $step84 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $spm_mfcc_pitch $lang $sdir $alidir || exit 1
fi
# make align
latdir=$sdir/ali_train-lat
if [ ! -z $step85 ]; then
  steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj \
  $spm_mfcc_pitch $lang  $sdir $latdir || exit 1
  rm $latdir/fsts.*.gz 2>/dev/null
fi
# prepare mfcc-hires feature
sp_mfcc_hires=$datadir/acumon-train/mfcc-hires-sp
if [ ! -z $step86 ]; then
  for x in /home3/zpz505/w2018/acumen/acumon-d5-final/data/train/mfcc-pitch; do
     sdata=$datadir/acumon-train
     utils/copy_data_dir.sh $x $sdata
     utils/data/perturb_data_dir_speed_3way.sh $sdata  $sp_mfcc_hires
     feat=$sdata/feat/mfcc-hires-sp/data log=$sdata/feat/mfcc-hires-sp/log
     steps/make_mfcc.sh --cmd "$cmd"  --nj $nj --mfcc-config conf/mfcc_hires.conf \
     $sp_mfcc_hires $log $feat || exit 1
     steps/compute_cmvn_stats.sh $sp_mfcc_hires $log $feat || exit 1
     utils/fix_data_dir.sh $sp_mfcc_hires
  done
fi
# combine data
spm_mfcc_hires=$datadir/acumon-train/spm-mfcc-hires
if [ ! -z $step87 ]; then
  utils/combine_data.sh $spm_mfcc_hires \
  seame2019/data/i2r-rasp-seame/mfcc_hires_sp \
  $sp_mfcc_hires
fi
# train online ivector extractor
tgtdir=acumon-seame-i2r-raspi/exp
if [ ! -z $step88 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
   --splice-opts "--left-context=3 --right-context=3" \
  --max-utts 200000 --subsample 2 \
  $spm_mfcc_hires \
  $tgtdir/pca_transform
fi
# train the diagonal UBM
if [ ! -z $step89 ]; then
   steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj $nj \
    --num-frames 15000000 \
    --num-threads 10 \
    $spm_mfcc_hires 512 \
    $tgtdir/pca_transform $tgtdir/diag_ubm
fi
# train ivector extractor
if [ ! -z $step90 ]; then
   steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj $nj \
  --num-processes 1 \
    $spm_mfcc_hires  $tgtdir/diag_ubm  $tgtdir/ivector-extractor || exit 1;
fi

# make ivectors for the training data
ivector_train=$tgtdir/ivector-train
if [ ! -z $step91 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
  $spm_mfcc_hires ${spm_mfcc_hires}-max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
  ${spm_mfcc_hires}-max2 \
  $tgtdir/ivector-extractor $ivector_train
fi

# make lang-sil
oldlang=seame2019/data/lang_base
olddictdir=seame2019/data/local/dict
dictdir=seame2019/data/local/dict-silprob
lang=seame2019/data/lang_base-silprob
if [ ! -z $step92 ]; then
  steps/get_prons.sh --cmd "$cmd"  $spm_mfcc_hires $oldlang $alidir || exit 1;
  utils/dict_dir_add_pronprobs.sh --max-normalize true \
  $olddictdir $alidir/pron_counts_nowb.txt $alidir/sil_counts_nowb.txt \
  $alidir/pron_bigram_counts_nowb.txt $dictdir || exit 1;
  [[ -f "$dictdir/lexiconp.txt" ]] && rm $dictdir/lexiconp.txt
  utils/validate_dict_dir.pl $dictdir || exit 1;
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang || exit 1;
  [ -f $oldlang/G.fst ] && cp $oldlang/G.fst $lang/
fi

chain_lang=$tgtdir/chain_lang
if [ ! -z $step93 ]; then
  source-scripts/egs/ngaho-sg-english-i2r-2018/make_chain_lang.sh \
  $lang $chain_lang
fi
# build decision tree for chain lang 
treedir=$tgtdir/chain_tree
if [ ! -z $step94 ]; then
    steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$cmd" 14000  $spm_mfcc_pitch  \
    $chain_lang $alidir  $treedir || exit 1
fi

# train chain models
dir=$tgtdir/chain-tdnn-f
[ -d $dir ] || mkdir -p $dir
if [ ! -z $step95 ]; then
    
  num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | /usr/bin/python2)
  tdnn_opts="l2-regularize=0.01 dropout-proportion=0.0 dropout-per-dim-continuous=true"
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"

  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  input dim=100 name=ivector
  input dim=40 name=input

  # please note that it is important to have input layer with the name=input
  # as the layer immediately preceding the fixed-affine-layer to enable
  # the use of short notation for the descriptor
  fixed-affine-layer name=lda input=Append(-1,0,1,ReplaceIndex(ivector, t, 0)) affine-transform-file=$dir/configs/lda.mat

  # the first splicing is moved before the lda layer, so no splicing here
  relu-batchnorm-dropout-layer name=tdnn1 $tdnn_opts dim=1024
  tdnnf-layer name=tdnnf2 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf3 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf4 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=1
  tdnnf-layer name=tdnnf5 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=0
  tdnnf-layer name=tdnnf6 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf7 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf8 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf9 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf10 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf11 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf12 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  tdnnf-layer name=tdnnf13 $tdnnf_opts dim=1024 bottleneck-dim=128 time-stride=3
  linear-component name=prefinal-l dim=192 $linear_opts


  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF
  steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/

fi
train_ivector_dir=$ivector_train
train_mfcc_hires=$spm_mfcc_hires
num_jobs_inital=3
num_jobs_final=16
if [ ! -z $step96 ]; then
    steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
    --feat.online-ivector-dir=$train_ivector_dir \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=10 \
    --trainer.frames-per-iter=5000000 \
    --trainer.optimization.num-jobs-initial=$num_jobs_inital \
    --trainer.optimization.num-jobs-final=$num_jobs_final \
    --trainer.optimization.initial-effective-lrate=0.0005 \
    --trainer.optimization.final-effective-lrate=0.00005 \
    --trainer.num-chunk-per-minibatch=128,64 \
    --trainer.optimization.momentum=0.0 \
    --egs.chunk-width=$chunk_width \
    --egs.chunk-left-context=0 \
    --egs.chunk-right-context=0 \
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

# make graph
chain_lang_test=${chain_lang}-test
[ -d $chain_lang_test ] || mkdir -p $chain_lang_test
graphdir=$dir/graph
if [ ! -z $step101 ]; then
  cp seame2019/exp/chain_lang-test/G.fst $chain_lang_test/
  cp -rL $chain_lang/* $chain_lang_test/
  utils/mkgraph.sh $chain_lang_test $dir  $graphdir
fi
# extract ivector for the two data sets
if [ ! -z $step102 ]; then
  for x in {sge,man}; do
    data=seame2019/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/ivector-dev-${x}
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
    $data $tgtdir/ivector-extractor $dev_ivector
  done
fi
# decode & evaluate

if [ ! -z $step103 ]; then
  for x in sge; do
    data=seame2019/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/ivector-dev-${x}
    decode_dir=$dir/decode-dev-${x}
    steps/nnet3/decode.sh --nj 10 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir $dev_ivector \
    $graphdir $data  $decode_dir  || exit 1
  done
fi

if [ ! -z $step104 ]; then
  for x in man; do
    data=seame2019/data/dev-${x}/mfcc-hires
    dev_ivector=$tgtdir/ivector-dev-${x}
    decode_dir=$dir/decode-dev-${x}
    steps/nnet3/decode.sh --nj 10 \
    --acwt 1.0 --post-decode-acwt 10.0 \
    --online-ivector-dir $dev_ivector \
    $graphdir $data  $decode_dir  || exit 1
  done
fi

