#!/bin/bash

# By Haihua Xu, TL@NTU, 2019

echo 
echo "$0 $@"
echo

. path.sh
set -e
# begin option
cmd="slurm.pl --quiet"
steps=
nj=120
srcdict=/home4/hhx502/w2019/exp/dec-02-2018-telium-data/data/local/dict/lexicon.txt
train_stage=-10
egs_stage=-100
xent_regularize=0.1
dropout_schedule='0,0@0.20,0.5@0.50,0'
tts_expdir=no-tone-bnf-tri3ali-head10
# training chunk-options
chunk_width=140,100,160
# we don't need extra left/right context for TDNN systems.
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=true
common_egs_dir=""

# end option

function UsageExample {
 cat <<EOF
 
 $0 --steps 1 --cmd "$cmd" --nj $nj \
 --srcdict $srcdict \
 /data/users/ngaho/corpora/Klass-SgEnglish \
 /data/users/ngaho/corpora/MML_Recording \
 /home4/hhx502/w2019/exp/may_09_klass_d3 

# step01 -- step10: demo data preparations
# step11 -- step12: gmm-hmm training
# step13 -- step14: build language models & finite-state transducer grammar
# step15 -- step18: train ivector extractor to extract ivector features as input to train tdnnf
# step20 -- step23: train tdnnf
# step24 -- step26: do evaluation on tdnnf
# step27 -- step30: train pronunciation probability lexicon, i.e., lexicon modeling
# step31 -- step41: tuning the tdnnf models with different methods
# step42 -- step47: demo using recurrent neural language models to rescore lattice 
# step48 -- step48: demo using more text data to build conventional n-gram language models
# step49 -- step50: demo using g2p to add more pronunciations to the lexicon

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

srctrain=$1
srcdev=$2
expdir=$3

# make kaldi data
data=$expdir/data/local/train
[ -d $data ] || mkdir -p $data
if [ ! -z $step01 ]; then
  [ -f $data/wav.scp ] || \
      { find $srctrain -name "*.wav" | \
      ./source-scripts/egs/w2019/klass_d3/make_wavscp.pl > $data/wav.scp
  }
  [ -f $data/text_grid.txt ]  || \
      find $srctrain -name "*.TextGrid"  > $data/text_grid.txt
  ./source-scripts/egs/w2019/klass_d3/textgrid-cracker.pl --convert-utf16 $data/text_grid.txt ""  $data | \
  ./source-scripts/egs/w2019/klass_d3/modify_wavid.pl | \
  ./source-scripts/egs/w2019/klass_d3/filter_utterance.pl | dos2unix > $data/transferred_text_grid.txt 
  cat $data/transferred_text_grid.txt | \
      ./source-scripts/egs/w2019/klass_d3/make_kaldi_data.pl  $data 
  utils/utt2spk_to_spk2utt.pl < $data/utt2spk > $data/spk2utt
  utils/fix_data_dir.sh $data
  echo  "## LOG (step01): done with '$data'"
fi
# check out-of-vocabulary statistics given a reference dictionary
if [ ! -z $step02 ]; then
  cut -d' ' -f2- $data/text | \
      ./source-scripts/egs/w2019/klass_d3/show-oov-with-count.pl --from=1 $srcdict > $data/oov-count.txt
  [ -f $data/oov-transfer-dict.txt ] || \
      {
      cut -f1 $data/oov-count.txt | \
	  perl -ane 'chomp; print "$_\t$_\n";' > $data/oov-transfer-dict.txt
  }
  [ -f $data/text-not-transfered ] || \
      cp $data/text $data/text-not-transfered
  cat $data/text-not-transfered | \
      ./source-scripts/egs/w2019/klass_d3/transfer-utterance-with-dict-ignore-oov.pl\
    $data/oov-transfer-dict.txt > $data/text
  cat $data/text | \
      ./source-scripts/egs/w2019/klass_d3/show-oov-with-count.pl --from=2 \
      $srcdict > $data/oov-count02.txt
  echo "## LOG (step02): done with '$data'"
fi

# now we are preparing test data
data=$expdir/data/local/test
[ -d $data ] || mkdir -p $data
if [ ! -z $step03 ]; then
  [ -f $data/wav.scp ] || \
      { find $srcdev/audio -name "*.wav" | \
      ./source-scripts/egs/w2019/klass_d3/make_wavscp.pl > $data/wav.scp
  }
  [ -f $data/text_grid.txt ]  || \
      find $srcdev/ref/ -name "*.TextGrid"  > $data/text_grid.txt
  ./source-scripts/egs/w2019/klass_d3/textgrid-cracker.pl  $data/text_grid.txt ""  $data | \
      ./source-scripts/egs/w2019/klass_d3/modify_wavid.pl | \
      ./source-scripts/egs/w2019/klass_d3/filter_utterance.pl | dos2unix > $data/transferred_text_grid.txt 
  cat $data/transferred_text_grid.txt | \
      ./source-scripts/egs/w2019/klass_d3/make_kaldi_data.pl  $data 
  utils/utt2spk_to_spk2utt.pl < $data/utt2spk > $data/spk2utt
  utils/fix_data_dir.sh $data
  echo "## LOG ($0): done with '$data'"
fi

# check out-of-vocabulary statistics given a reference dictionary
if [ ! -z $step04 ]; then
  cut -d' ' -f2- $data/text | \
      ./source-scripts/egs/w2019/klass_d3/show-oov-with-count.pl --from=1 $srcdict > $data/oov-count.txt
  [ -f $data/oov-transfer-dict.txt ] || \
      {
      cut -f1 $data/oov-count.txt | \
	  perl -ane 'chomp; print "$_\t$_\n";' > $data/oov-transfer-dict.txt
  }
  [ -f $data/text-not-transfered ] || \
      cp $data/text $data/text-not-transfered
  cat $data/text-not-transfered | \
      ./source-scripts/egs/w2019/klass_d3/transfer-utterance-with-dict-ignore-oov.pl\
    $data/oov-transfer-dict.txt > $data/text
  cat $data/text | \
      ./source-scripts/egs/w2019/klass_d3/show-oov-with-count.pl --from=2 \
      $srcdict > $data/oov-count02.txt
  echo "## LOG ($0): done with '$data'"
fi
# now, we are going to divide training data into two parts
# one is for regular training, the other is for fine-tuning.
# then we divid the fine-tuning part into two parts again,
# one is for real fine-tuning, the other is to test the effectiveness 
# of fine-tuning. 
tune_overall=$expdir/data/local/tune_overall
train=$expdir/data/local/train_train
[ -d $train ] || mkdir -p $train
[ -d $tune_overall ] || mkdir -p $tune_overall
if [ ! -z $step05 ]; then
  cat $expdir/data/local/train/utt2spk | \
      perl -ane 'chomp; @A = split(/\s+/); print "$A[1]\n";' | \
      sort -u | utils/shuffle_list.pl --srand 777 | \
      head -100 > $tune_overall/spklist
   utils/subset_data_dir.sh --spk-list $tune_overall/spklist \
       $expdir/data/local/train $tune_overall
    cat $expdir/data/local/train/utt2spk | \
      perl -ane 'chomp; @A = split(/\s+/); print "$A[1]\n";' | \
      sort -u | utils/shuffle_list.pl --srand 777 | \
      tail -n +101 > $train/spklist
    utils/subset_data_dir.sh --spk-list $train/spklist \
	 $expdir/data/local/train $train
fi
# now we define tune and tune_test data sets
# this time we are going to use utterance number to split data
tune_train=$expdir/data/local/tune_train
tune_test=$expdir/data/local/tune_test
[ -d $tune_train ] || mkdir -p $tune_train
[ -d $tune_test ] || mkdir -p $tune_test
if [ ! -z $step06 ]; then
  cat $tune_overall/utt2spk | \
      perl -ane 'chomp; @A = split(/\s+/); print "$A[0]\n";' | \
      sort -u | utils/shuffle_list.pl --srand 777 | \
      head -2000 > $tune_test/uttlist
  utils/subset_data_dir.sh --utt-list $tune_test/uttlist \
      $tune_overall $tune_test
  cat $tune_overall/utt2spk | \
      perl -ane 'chomp; @A = split(/\s+/); print "$A[0]\n";' | \
      sort -u | utils/shuffle_list.pl --srand 777 | \
      tail -n +2001 > $tune_train/uttlist
  utils/subset_data_dir.sh --utt-list $tune_train/uttlist \
      $tune_overall $tune_train
fi
# now copy all  data sets to the data folder 
if [ ! -z $step07 ]; then
  cp -rL $train  $expdir/data/train
  cp -rL $tune_train $expdir/data/tune_train
  cp -rL $tune_test $expdir/data/tune_test
  cp -rL $expdir/data/local/test $expdir/data/test
fi

# now prepare the lexicon. We are doing this by:
# extract a subset lexicon from our source lexicon to cover
# the training data
dictdir=$expdir/data/local/dict
[ -d $dictdir ] || mkdir -p $dictdir
if [ ! -z $step08 ]; then
  cut -d' ' -f2- $data/text | \
      ./source-scripts/egs/w2019/klass_d3/subset_dict.pl $srcdict | \
   cat - <(grep '<' $srcdict) | sort -u > $dictdir/lexicon.txt
  grep -v '<' $dictdir/lexicon.txt | \
      ./source-scripts/egs/w2019/klass_d3/print_phone_set.pl > $dictdir/nonsilence_phones.txt
   cp $(dirname $srcdict)/{extra_questions.txt,silence_phones.txt,optional_silence.txt} $dictdir/
   utils/validate_dict_dir.pl $dictdir
  echo "## LOG (step03): done with '$dictdir'"
fi
# prepare lang
langdir=$expdir/data/lang_base
[ -d $langdir ] || mkdir -p $langdir
if [ ! -z $step09 ]; then
    utils/prepare_lang.sh $dictdir "<unk>" $langdir/tmp $langdir
fi

# now make feature to train gmm-hmm models
if [ ! -z $step10 ]; then
    for  x in $expdir/data/{train,tune_train,tune_test,test}; do
    data=$x/mfcc-pitch; log=$x/feat/mfcc-pitch/log; feat=$x/feat/mfcc-pitch/data
    utils/data/copy_data_dir.sh $x $data
    steps/make_mfcc_pitch.sh --mfcc-config ./conf/mfcc.conf --cmd "$cmd"  --nj $nj \
    --pitch-config ./conf/pitch.conf \
    $data $log $feat || exit 1
    steps/compute_cmvn_stats.sh $data $log $feat || exit 1 
    utils/fix_data_dir.sh $data 
  done
fi

# we are going to train gmm-hmm models
gmmdir=$expdir/exp
[ -d $gmmdir ] || mkdir -p $gmmdir
if [ ! -z $step11 ]; then
  echo "## LOG (step11): train gmm-hmm"
  ./source-scripts/egs/w2019/klass_d3/train_gmm.sh --cmd "$cmd" --nj $nj \
      --steps 1-4 \
      --state-num 3000 \
      --pdf-num 60000\
   $expdir/data/train/mfcc-pitch $langdir $gmmdir || exit 1
fi
# make phone lattice preparing for tdnn training
sdir=$gmmdir/tri3a
latdir=$sdir/ali_train_lat
data=$expdir/data/train/mfcc-pitch
if [ ! -z $step12 ]; then
    steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj \
  $data $langdir  $sdir $latdir || exit 1  
fi
# build language model using transcript from training data
lmdir=$expdir/data/local/lm
[ -d $lmdir ] || mkdir -p $lmdir
if [ ! -z $step13 ]; then 
  grep  -vP '#|<eps>|<s>|</s>' $langdir/words.txt | awk '{print $1; }' | gzip -c > $lmdir/vocab.gz   
  cut -d' ' -f2- $expdir/data/train/text | gzip -c > $lmdir/train-text.gz 
  cut -d' ' -f2- $expdir/data/test/text | gzip -c > $lmdir/dev-text.gz
  ngram-count -order 3 -kndiscount -interpolate \
  -vocab $lmdir/vocab.gz  -unk -sort -text $lmdir/train-text.gz -lm $lmdir/lm3.gz
  ngram -order 3 -lm $lmdir/lm3.gz  -ppl $lmdir/dev-text.gz 
  echo "## LOG (step13): language model training is done, check '$lmdir'"
fi
# convert arpa lm to fst grammar
if [ ! -z $step14 ]; then
  ./source-scripts/egs/w2019/klass_d3/arpa2G.sh $lmdir/lm3.gz $langdir $langdir
fi 
# we are making higher resolution mfcc features for
# ivector extractor and tdnn training
if [ ! -z $step15 ]; then
    for  x in $expdir/data/{train,tune_train,tune_test,test}; do
    data=$x/mfcc-hires; log=$x/feat/mfcc-hires/log; feat=$x/feat/mfcc-hires/data
    utils/data/copy_data_dir.sh $x $data
    steps/make_mfcc.sh --mfcc-config ./conf/mfcc_hires.conf --cmd "$cmd"  --nj $nj \
    $data $log $feat || exit 1
    steps/compute_cmvn_stats.sh $data $log $feat || exit 1 
    utils/fix_data_dir.sh $data 
  done
fi
#
# now we are trining ivector extractor
# first, we train transform
ivector_extractor_dir=$expdir/exp
data=$expdir/data/train/mfcc-hires
if [ ! -z $step16 ]; then
    steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
  --splice-opts "--left-context=3 --right-context=3" \
  --max-utts 2000000 --subsample 2 \
  $data \
  $ivector_extractor_dir/pca_transform 
fi
# train the diagonal UBM, collecting statistics for ivector extractor
if [ ! -z $step17 ]; then
     steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj $nj \
    --num-frames 500000 \
    --num-threads 10 \
    $data 512 \
    $ivector_extractor_dir/pca_transform $ivector_extractor_dir/diag_ubm
fi
# train ivector extractor
if [ ! -z $step18 ]; then
     steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj $nj \
	 --num-processes 1 \
    $data  $ivector_extractor_dir/diag_ubm  $ivector_extractor_dir/ivector-extractor || exit 1;
fi

# generate ivector feature for the training data
# to train tdnn models
ivector_train=$ivector_extractor_dir/ivector-train
if [ ! -z $step19 ]; then
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
  $data ${data}-max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
  ${data}-max2 \
  $ivector_extractor_dir/ivector-extractor $ivector_train
fi
# change hmm topology
# each hmm has single state
# this can reduce the duration of hmm to only one frame
# please refer to Dan's paper
dir=$ivector_extractor_dir
chain_lang=$dir/chain_lang
if [ ! -z $step20 ]; then
  source-scripts/egs/w2019/klass_d3/make_chain_lang.sh $langdir $chain_lang 
fi
# buil decision tree for chain lang
data=$expdir/data/train/mfcc-pitch
treedir=$dir/chain_tree
alidir=$expdir/exp/tri3a/ali_train
if [ ! -z $step21 ]; then
    steps/nnet3/chain/build_tree.sh \
    --frame-subsampling-factor 3 \
    --context-opts "--context-width=2 --central-position=1" \
    --cmd "$cmd" 3000  $data  \
    $chain_lang $alidir  $treedir || exit 1
fi
# now define the tdnn topology
dir=$ivector_extractor_dir/tdnnf
if [ ! -z $step22 ]; then
    
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

# train tdnnf
train_ivector_dir=$ivector_train
train_mfcc_hires=$expdir/data/train/mfcc-hires
if [ ! -z $step23 ]; then
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
    --trainer.optimization.num-jobs-final=8 \
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
# now, we are testing the tdnnf model
graphdir=$dir/graph_base
if [ ! -z $step24 ]; then
  utils/mkgraph.sh $chain_lang $dir  $graphdir
fi
# extract ivector feature for two test sets
# and decode them to evaluat the performance
ivector_extractor=$ivector_extractor_dir/ivector-extractor
if [ ! -z $step25 ]; then
  for x in test tune_test; do
      data=$expdir/data/$x/mfcc-hires
      dev_ivector=$ivector_extractor_dir/ivector-data-$x
      steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
	  $data  $ivector_extractor  $dev_ivector || exit 1
  done
fi
# decode
if [ ! -z $step26 ]; then
    for x in tune_test; do
	data=$expdir/data/$x/mfcc-hires
	dev_ivector=$ivector_extractor_dir/ivector-data-$x
	decode_dir=$dir/decode-data-$x
	steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	    --acwt 1.0 --post-decode-acwt 10.0 \
	    --online-ivector-dir $dev_ivector \
	    $graphdir $data  $decode_dir  || exit 1 
    done
fi

# now we are take advantage of ASR alignment to 
# learn pronunciation probability, hopefully yielding
# an improved lexicon
srcdict=$expdir/data/local/dict
tgtdict=$expdir/data/local/pdict
alidir=$expdir/exp/tri3a/ali_train
[ -d $tgtdict ] || mkdir -p $tgtdict
if [ ! -z $step27 ]; then
  steps/get_prons.sh $expdir/data/train/mfcc-pitch \
      $expdir/data/lang_base  $alidir
  utils/dict_dir_add_pronprobs.sh $srcdict \
      $alidir/pron_counts_nowb.txt $alidir/sil_counts_nowb.txt \
      $alidir/pron_bigram_counts_nowb.txt $tgtdict
  
fi
# create new lang using the new lexicon with pronunciation probability
langdir=$expdir/data/lang_prob_pron
if [ ! -z $step28 ]; then
  utils/prepare_lang.sh $tgtdict "<unk>" $langdir/tmp $langdir
fi
# make graph again to evaluate the new lexicon
graphdir=$dir/graph_prob_pron
if [ ! -z $step29 ]; then
  cp $expdir/data/lang_base/G.fst $langdir/
  utils/mkgraph.sh $langdir $dir  $graphdir
fi
# evaluate on the test data sets
if [ ! -z $step30 ]; then
    for x in test tune_test; do
	data=$expdir/data/$x/mfcc-hires
	dev_ivector=$ivector_extractor_dir/ivector-data-$x
	decode_dir=$dir/decode-data-${x}-prob-pron
	steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	    --acwt 1.0 --post-decode-acwt 10.0 \
	    --online-ivector-dir $dev_ivector \
	    $graphdir $data  $decode_dir  || exit 1
    done
fi

# now we are trying to play with transfer learning
# we are going to train a new gmm-hmm model
# to simulate the real transfer learning case
data=$expdir/data/tune_train/mfcc-pitch
sdir=$expdir/exp/tri3a
alidir=$sdir/ali_tune_train
tune_sdir=$expdir/exp/tri3a_tune
tune_alidir=$tune_sdir/ali_train
if [ ! -z $step31 ]; then
  if false; then
      steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
	  $data $langdir $sdir $alidir || exit 1
  fi
  steps/train_sat.sh --cmd "$cmd" \
  3000 60000 $data $langdir  $alidir   $tune_sdir || exit 1

  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
      $data $langdir $tune_sdir $tune_alidir || exit 1
fi
# now, we are doing transfer learning
latdir_tune=$sdir/ali_tune_train_lat
if [ ! -z $step32 ]; then
    steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj \
  $data $langdir  $sdir $latdir_tune || exit 1  
fi
ivector_train=$expdir/exp/tl_shared_tree/ivector-train
data=$expdir/data/tune_train/mfcc-hires
train_mfcc_hires=$data
if [ ! -z $step33 ]; then
   utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
  $data ${data}-max2

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $nj \
  ${data}-max2 \
  $ivector_extractor_dir/ivector-extractor $ivector_train  
fi
sdir=$dir
dir=$expdir/exp/tl_shared_tree/tdnnf
phone_lm_scales=1,5
if [ ! -z $step34 ]; then
    num_targets=$(tree-info $treedir/tree |grep num-pdfs|awk '{print $2}')
  learning_rate_factor=$(echo "print 0.5/$xent_regularize" | /usr/bin/python2)
  tdnnf_opts="l2-regularize=0.01 dropout-proportion=0.0 bypass-scale=0.66"
  linear_opts="l2-regularize=0.01 orthonormal-constraint=-1.0"
  prefinal_opts="l2-regularize=0.01"
  output_opts="l2-regularize=0.005"
  primary_lr_factor=0.25
  mkdir -p $dir/configs
  cat <<EOF > $dir/configs/network.xconfig
  prefinal-layer name=prefinal-chain input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output include-log-softmax=false dim=$num_targets $output_opts

  prefinal-layer name=prefinal-xent input=prefinal-l $prefinal_opts big-dim=1024 small-dim=192
  output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor $output_opts
EOF

  steps/nnet3/xconfig_to_configs.py --existing-model $sdir/final.mdl \
    --xconfig-file  $dir/configs/network.xconfig  \
    --config-dir $dir/configs/
 $cmd $dir/log/generate_input_mdl.log \
    nnet3-copy --edits="set-learning-rate-factor name=* learning-rate-factor=$primary_lr_factor" $sdir/final.mdl - \| \
      nnet3-init --srand=1 - $dir/configs/final.config $dir/input.raw  || exit 1;
  # update phone lm
   echo "$0: compute {den,normalization}.fst using weighted phone LM with wsj and rm weight $phone_lm_scales."
  steps/nnet3/chain/make_weighted_den_fst.sh --cmd "$cmd" \
    --num-repeats $phone_lm_scales \
    --lm-opts '--num-extra-lm-states=200' \
    $treedir $tune_alidir $dir || exit 1;
fi
# tune the tdnnf model
[ $train_stage -lt -4 ] || train_stage=-4
if [ ! -z $step35 ]; then
     mv  $treedir/num_jobs   $treedir/old_num_jobs
     cat $latdir_tune/num_jobs > $treedir/num_jobs
      steps/nnet3/chain/train.py --stage=$train_stage \
    --cmd="$cmd" \
	  --trainer.input-model $dir/input.raw \
    --feat.online-ivector-dir=$ivector_train \
    --feat.cmvn-opts="--norm-means=false --norm-vars=false" \
    --chain.xent-regularize $xent_regularize \
    --chain.leaky-hmm-coefficient=0.1 \
    --chain.l2-regularize=0.0 \
    --chain.apply-deriv-weights=false \
    --chain.lm-opts="--num-extra-lm-states=2000" \
    --trainer.dropout-schedule $dropout_schedule \
    --trainer.srand=$srand \
    --trainer.max-param-change=2.0 \
    --trainer.num-epochs=5 \
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
    --lat-dir=$latdir_tune \
    --dir=$dir 
     mv $treedir/old_num_jobs > $treedir/num_jobs
fi

# now evaluate the performance of the tuned tdnnf models
# decode
# since the tree is not changed, the graph is not changed either.
if [ ! -z $step36 ]; then
    for x in tune_test; do
	data=$expdir/data/$x/mfcc-hires
	dev_ivector=$ivector_extractor_dir/ivector-data-$x
	decode_dir=$dir/decode-data-$x
	steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	    --acwt 1.0 --post-decode-acwt 10.0 \
	    --online-ivector-dir $dev_ivector \
	    $graphdir $data  $decode_dir  || exit 1 
    done
fi
# we also test another recipe for transfer learning
tgtdir=./exp/may_09_klass_d3/exp/transfer_learning
langdir=$tgtdir/lang
dir=$tgtdir/tdnnf
graphdir=$dir/graph
if [ ! -z $step37 ]; then
   utils/mkgraph.sh $langdir $dir  $graphdir
fi
# do evaluation
if [ ! -z $step38 ]; then
    for x in tune_test; do
	data=$expdir/data/$x/mfcc-hires
	dev_ivector=$ivector_extractor_dir/ivector-data-$x
	decode_dir=$dir/decode-data-$x
	steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	    --acwt 1.0 --post-decode-acwt 10.0 \
	    --online-ivector-dir $dev_ivector \
	    $graphdir $data  $decode_dir  || exit 1 
    done
fi

# revisit transfer learning again
if [ ! -z $step39 ]; then
    if false; then
     ./source-scripts/egs/w2019/klass_d3/do-transfer-learning.sh --steps 1-6 --train-stage -10 --cmd "$cmd"
   fi
tgtdir=exp/may_09_klass_d3/exp/transfer-learning-3ksenone
langdir=$tgtdir/lang
dir=$tgtdir/tdnnf
graphdir=$dir/graph
if true; then
   utils/mkgraph.sh $langdir $dir  $graphdir
fi
# do evaluation
if true; then
    for x in tune_test; do
	data=$expdir/data/$x/mfcc-hires
	dev_ivector=$ivector_extractor_dir/ivector-data-$x
	decode_dir=$dir/decode-data-$x
	steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	    --acwt 1.0 --post-decode-acwt 10.0 \
	    --online-ivector-dir $dev_ivector \
	    $graphdir $data  $decode_dir  || exit 1 
    done
fi
fi
# 

# now we are testing recurrent neural network rescoring
dir=$expdir/exp/rnnlm
[ -d $dir ] || mkdir -p $dir
# prepare data
confdir=$dir/config
[ -d $confdir ] || mkdir -p $confdir
textdir=$dir/data
[ -d $textdir ] || mkdir -p $textdir
if [ ! -z $step42 ]; then
  cat $expdir/data/train/text $expdir/data/tune_train/text | \
      cut -d' ' -f2- | \
      utils/shuffle_list.pl --srand 777 | \
      awk -v  text_dir=$textdir '{if(NR%20 == 0) { print >text_dir"/dev.txt"; } else {print;}}' > $textdir/train.txt
fi
# prepare word list
wordlist=$langdir/words.txt
if [ ! -z $step43 ]; then
  cp $wordlist $confdir/
  n=`cat $dir/config/words.txt | wc -l`
  echo "<brk> $n" >> $confdir/words.txt
  echo "<unk>" >$confdir/oov.txt
  cat > $confdir/data_weights.txt <<EOF
train 10 1.0
EOF
   rnnlm/get_unigram_probs.py --vocab-file=$confdir/words.txt \
                             --unk-word="<unk>" \
                             --data-weights-file=$confdir/data_weights.txt \
       $textdir |awk 'NF==2' >$confdir/unigram_probs.txt
    # choose features
  rnnlm/choose_features.py --unigram-probs=$dir/config/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features=50000 \
                           --min-frequency 1.0e-03 \
                           --special-words='<s>,</s>,<brk>,<unk>' \
                           $dir/config/words.txt > $dir/config/features.txt
fi
# design recurrent neural network language models 
embedding_dim=800
lstm_rpd=200
lstm_nrpd=200
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=10
stage=-10
train_stage=-10

if [ ! -z $step44 ]; then
  lstm_opts="l2-regularize=$comp_l2"
  tdnn_opts="l2-regularize=$comp_l2"
  output_opts="l2-regularize=$output_l2"

  cat >$dir/config/xconfig <<EOF
input dim=$embedding_dim name=input
relu-renorm-layer name=tdnn1 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1)) 
fast-lstmp-layer name=lstm1 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn2 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-2))
fast-lstmp-layer name=lstm2 cell-dim=$embedding_dim recurrent-projection-dim=$lstm_rpd non-recurrent-projection-dim=$lstm_nrpd $lstm_opts
relu-renorm-layer name=tdnn3 dim=$embedding_dim $tdnn_opts input=Append(0, IfDefined(-1))
output-layer name=output $output_opts include-log-softmax=false dim=$embedding_dim
EOF
  rnnlm/validate_config_dir.sh $textdir $dir/config
fi
# prepare rnn lm dir
if [ ! -z $step45 ]; then
    # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 200.0 \
                             $textdir $dir/config $dir
fi
# do training 
if [ ! -z $step46 ]; then
    rnnlm/train_rnnlm.sh --num-jobs-initial 1 --num-jobs-final 1 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi

# test 
if [ ! -z $step47 ]; then
    for x in tune_test; do
	data=$expdir/data/$x/mfcc-hires
	decode_dir=$expdir/exp/tdnnf/decode-data-$x
	rnnlm/lmrescore_pruned.sh \
	    --cmd "$cmd --mem 10G" \
	    --weight 0.8 --max-ngram-order 3 \
	    $langdir $dir \
	    $data $decode_dir \
	    ${decode_dir}_rnnlm_rescore
    done  
fi

# now we demo updating conventional n-gram language models
# we are using tune_train transcript as extra text data
# ------------------------------------------------------------------- #
# Note: please do the following step-by-step for your real big data.  #
# ------------------------------------------------------------------- #
lmdir=$expdir/data/local/lm_tune_train
langdir_updated_g=$expdir/data/lang_updated_g
dir=$expdir/exp/tdnnf
graphdir=$dir/graph_updated_g
[ -d $lmdir ] || mkdir -p $lmdir
if [ ! -z $step48 ]; then
  grep  -vP '#|<eps>|<s>|</s>' $langdir/words.txt | awk '{print $1; }' | gzip -c > $lmdir/vocab.gz   
  cut -d' ' -f2- $expdir/data/train/text $expdir/data/tune_train/text | gzip -c > $lmdir/train-text.gz 
  cut -d' ' -f2- $expdir/data/tune_test/text | gzip -c > $lmdir/dev-text.gz
  # build tri-gram lm
  ngram-count -order 3 -kndiscount -interpolate \
  -vocab $lmdir/vocab.gz  -unk -sort -text $lmdir/train-text.gz -lm $lmdir/lm3.gz
  # test perplexity
  ngram -order 3 -lm $lmdir/lm3.gz  -ppl $lmdir/dev-text.gz 
  # update lang with the new grammar G
  ./source-scripts/egs/w2019/klass_d3/arpa2G.sh $lmdir/lm3.gz $langdir $langdir_updated_g
  # make decoding graph
   utils/mkgraph.sh $langdir_updated_g $dir  $graphdir
  # do evaluation
   for x in tune_test; do
       data=$expdir/data/$x/mfcc-hires
       dev_ivector=$ivector_extractor_dir/ivector-data-$x
       decode_dir=$dir/decode-data-${x}_updated_g
       steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	   --acwt 1.0 --post-decode-acwt 10.0 \
	   --online-ivector-dir $dev_ivector \
	   $graphdir $data  $decode_dir  || exit 1 
    done
fi
# now we demo updating lexicon by adding more lexicon
# we are training grapheme to phoneme models to
# label the new words
# Note: please to the following step-by-step
g2pdir=$expdir/exp/g2p
lexdir=$expdir/data/local/dict
[ -d $g2pdir ] || mkdir -p $g2pdir
g2pdictdir=$expdir/data/local/dict_g2p
g2p_langdir=$expdir/data/lang_g2p
[ -d $g2pdictdir ] || mkdir -p $g2pdictdir
if [ ! -z $step49 ]; then
  if [ ! -f $g2pdir/model-3 ]; then
      ./source-scripts/egs/w2019/klass_d3/train-g2p-model.sh --steps 1,2,3,4 \
	  $lexdir/lexicon.txt $g2pdir
  fi
# we use the following word list to test
  cat > $g2pdir/wordlist.txt <<EOF
townsfolks
toponymics
videographers
paramesvari
routings
platoons
dislikeable
disorientate
adreno
berumahtangga
EOF
# label lexicon
  if [ ! -z $g2pdictdir/g2p-lexicon.txt ]; then
    g2p.py --model $g2pdir/model-3 \
	--apply $g2pdir/wordlist.txt > $g2pdictdir/g2p-lexicon.txt
  fi
 cp $expdir/data/local/dict/* $g2pdictdir/
 rm $g2pdictdir/lexiconp.txt 2>/dev/null
 cat $g2pdictdir/g2p-lexicon.txt >> $g2pdictdir/lexicon.txt
   utils/validate_dict_dir.pl $g2pdictdir
   utils/prepare_lang.sh $g2pdictdir "<unk>" $g2p_langdir/tmp $g2p_langdir  
fi
# update the g2p lang and do evaluation test again

g2p_lmdir=$expdir/data/local/lm_g2p
[ -d $g2p_lmdir ] || mkdir -p $g2p_lmdir
graphdir=$dir/graph_g2p
if [ ! -z $step50 ]; then
    grep  -vP '#|<eps>|<s>|</s>' $g2p_langdir/words.txt | awk '{print $1; }' | gzip -c > $g2p_lmdir/vocab.gz   
  cut -d' ' -f2- $expdir/data/train/text $expdir/data/tune_train/text | gzip -c > $g2p_lmdir/train-text.gz 
  cut -d' ' -f2- $expdir/data/tune_test/text | gzip -c > $g2p_lmdir/dev-text.gz
  # build tri-gram lm
  ngram-count -order 3 -kndiscount -interpolate \
  -vocab $g2p_lmdir/vocab.gz  -unk -sort -text $g2p_lmdir/train-text.gz -lm $g2p_lmdir/lm3.gz
  # test perplexity
  ngram -order 3 -lm $g2p_lmdir/lm3.gz  -ppl $g2p_lmdir/dev-text.gz 
  # update lang with the new grammar G
  ./source-scripts/egs/w2019/klass_d3/arpa2G.sh $g2p_lmdir/lm3.gz $g2p_langdir $g2p_langdir
  # make decoding graph
   utils/mkgraph.sh $g2p_langdir  $dir  $graphdir
   for x in tune_test; do
       data=$expdir/data/$x/mfcc-hires
       dev_ivector=$ivector_extractor_dir/ivector-data-$x
       decode_dir=$dir/decode-data-${x}_g2p
       steps/nnet3/decode.sh --nj $nj --cmd "$cmd" \
	   --acwt 1.0 --post-decode-acwt 10.0 \
	   --online-ivector-dir $dev_ivector \
	   $graphdir $data  $decode_dir  || exit 1 
    done
fi

echo "## LOG ($0): Done so far so good !" && exit 0
