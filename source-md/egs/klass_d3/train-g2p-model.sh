#!/bin/bash

. path.sh
. cmd.sh

# begin options
steps=
size_constraints="1,2,1,2"
dev_ratio="5%"
# end options
echo 
echo LOG: $0 $@
echo 

. utils/parse_options.sh || exit 1
function Usage {
  cat<<END

 $(basename $0) [options] <lexicon> <dir>
 [options]:
 --steps                                  # value, "$steps"
 --size-contraints                        # value, "$size_constraints"
 --dev-ratio                              # value, "$dev_ratio"
 [steps]:
 1: normalize lexicon
 2: train unigram model
 3: train bigram model
 4: train trigram model

END
}
if [ $# -ne 2 ]; then
  Usage && exit 1
fi
lexicon=$1
dir=$2

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    declare step$index=1
  done
fi

g2p=$(which g2p.py)
if [ -z $g2p ]; then
  echo "ERROR, g2p is not installed" && exit 1
fi 

if [ ! -z $step01 ]; then
  tmpdir=$dir/temp
  [ -d $tmpdir ] || mkdir -p $tmpdir
  grep -v '^[^a-z]' $lexicon > $tmpdir/lex.txt
fi

lex=$tmpdir/lex.txt 
if [ ! -f $lex ]; then
  lex=$lexicon
fi
logdir=$dir/log
[ -d $logdir ] || mkdir -p $logdir
if [ ! -z $step02 ]; then
  echo "unigram training started @ `date`"
  $g2p  --encoding UTF-8 --train $lex \
  --devel $dev_ratio  -s $size_constraints --write-model $dir/model-1   > $logdir/unigram.log 2>&1 
  echo "Done @ `date`"
fi

if [ ! -z $step03 ]; then
  [ -f $dir/model-1 ] || \
  { echo "ERROR, model $dir/model-1 does not exist"; exit 1; }
  echo "bigram training started @ `date`"
  $g2p  --model $dir/model-1 --ramp-up --train $lex \
  --devel $dev_ratio  -s $size_constraints --write-model $dir/model-2  > $logdir/bigram.log 2>&1
  echo "Done @ `date`"
fi

if [ ! -z $step04 ]; then
  [ -f $dir/model-2 ] || \
  { echo "ERROR, model $dir/model-2 does not exist"; exit 1; }
  echo "trigram training started @ `date`"
  $g2p  --model $dir/model-2 --ramp-up --train $lex \
  --devel $dev_ratio  -s $size_constraints --write-model $dir/model-3 > $logdir/trigram.log 2>&1
  echo "Done @ `date`"
fi

if [ ! -z $step05 ]; then
  [ -f $dir/model-3 ] || \
  { echo "ERROR, model $dir/model-3 does not exist"; exit 1; }
  echo "trigram training started @ `date`"
  $g2p  --model $dir/model-3 --ramp-up --train $lex \
  --devel $dev_ratio  -s $size_constraints --write-model $dir/model-4 > $logdir/four_gram.log 2>&1
  echo "Done @ `date`"
fi
