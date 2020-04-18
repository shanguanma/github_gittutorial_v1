#!/bin/bash

#By maowang,2019@mipitalk

echo
echo "$0 $@"
echo

. path.sh

set -e

# start options
#cmd="slurm.pl --quiet --exclude=node01,node02,node06,node07,node08"
#cmd="run.pl"
steps=
nj=100
# end options

function UsageExample {
  cat <<EOF
    $0 --steps 1 --cmd "$cmd"  projects/wsj
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

if [ $# -ne 1 ]; then
  UsageExample && exit 1
fi

expdir=$1
langdir=$expdir/data/lang_base

# train rnnlm for rescoring
dir=$expdir/exp/rnnlm
[ -d $dir ] || mkdir -p $dir
# prepare data
confdir=$dir/config
[ -d $confdir ] || mkdir -p $confdir
textdir=$dir/data
[ -d $textdir ] || mkdir -p $textdir
if [ ! -z $step01 ]; then
  echo -n >$textdir/dev.txt
  cat $expdir/data/train/text | \
    cut -d' ' -f2- | \
    utils/shuffle_list.pl --srand 777 | \
    awk -v text_dir=$textdir '{if(NR%500 == 0) { print >text_dir"/dev.txt"; } else {print;}}' > $textdir/train.txt
fi
# prepare word list
wordlist=$langdir/words.txt
if [ ! -z $step02 ]; then
  cp $wordlist $confdir/
  n=`cat $confdir/words.txt | wc -l`
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
  rnnlm/choose_features.py --unigram-probs=$confdir/unigram_probs.txt \
                           --use-constant-feature=true \
                           --top-word-features=50000 \
                           --min-frequency 1.0e-04 \
                           --special-words='<s>,</s>,<brk>,<unk>' \
                           $confdir/words.txt > $confdir/features.txt
fi

# design recurrent neural network language models 
embedding_dim=800 #1600
lstm_rpd=200
lstm_nrpd=200
embedding_l2=0.001 # embedding layer l2 regularize
comp_l2=0.001 # component-level l2 regularize
output_l2=0.001 # output-layer l2 regularize
epochs=10 #20
stage=-10
train_stage=-10

if [ ! -z $step03 ]; then
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
  rnnlm/validate_config_dir.sh $textdir $confdir
fi
# prepare rnn lm dir
if [ ! -z $step04 ]; then
  # the --unigram-factor option is set larger than the default (100)
  # in order to reduce the size of the sampling LM, because rnnlm-get-egs
  # was taking up too much CPU (as much as 10 cores).
  rnnlm/prepare_rnnlm_dir.sh --unigram-factor 100.0 \
                             $textdir $confdir $dir
fi
# do training 
if [ ! -z $step05 ]; then
  rnnlm/train_rnnlm.sh --num-jobs-initial 2 --num-jobs-final 2 \
                       --embedding_l2 $embedding_l2 \
                       --stage $train_stage --num-epochs $epochs --cmd "$cmd" $dir
fi

# test backup:--weight 0.8
if [ ! -z $step06 ]; then
  for x in dev; do
    data=$expdir/data/$x
    decode_dir=$expdir/exp/tdnnf/decode-${x}-v1.0-large
    rnnlm/lmrescore_pruned.sh \
        --cmd "$cmd --mem 10G" \
        --weight 0.4 --max-ngram-order 3 \
        $langdir $dir \
        $data $decode_dir \
        ${decode_dir}-rnnlm-rescoring
  done
fi
