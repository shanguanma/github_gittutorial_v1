#!/bin/bash

. path.sh
. cmd.sh 

# begin sub
cmd='slurm.pl --quiet'
steps=
lmwt=9
beam=15
word_ins_penalty=0
decode_mbr=true
frame_shift=0.03
# begin sub

. parse_options.sh || exit 1


function Example {
 cat<<EOF

 $0 --steps 1 --cmd "$cmd" \
 ../acumen/update-sept-23-2017/exp/tdnn/chainname/decode-semi-data200 \
../acumen/update-sept-23-2017/exp/tdnn/chainname/final.mdl \
 ../acumen/update-sept-23-2017/data/lang-silprob 

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
if [ $# -ne 3 ]; then
  Example && exit 1
fi
latdir=$1
model=$2
lang=$3

dir=$latdir/ctm-conf-${lmwt}
[ -d $dir ] || mkdir -p $dir
nj=$(cat $latdir/num_jobs) || exit 1
if [ ! -z $step01 ]; then
  echo "## LOG (step01): started @ `date`"
  $cmd JOB=1:$nj $dir/scoring/log/get_ctm.JOB.log \
    set -e -o pipefail \; \
    mkdir -p $dir/score_$lmwt/ '&&' \
    lattice-scale --inv-acoustic-scale=$lmwt "ark:gunzip -c $latdir/lat.JOB.gz|" ark:- \| \
    lattice-add-penalty --word-ins-penalty=$word_ins_penalty ark:- ark:- \| \
    lattice-prune --beam=$beam ark:- ark:- \| \
    lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
    lattice-to-ctm-conf --frame-shift=$frame_shift  --decode-mbr=$decode_mbr ark:- - \| \
    utils/int2sym.pl -f 5 $lang/words.txt  \| tee $dir/score_$lmwt/utt.JOB.ctm
  echo "## LOG (step01): done @ `date`"
fi
if [ ! -z $step02 ]; then
  for x in $(seq 1 $nj); do
    cat $dir/score_$lmwt/utt.$x.ctm
  done  > $dir/score_$lmwt/utt.ctm
  echo "## LOG (step02): done with '$dir/score_$lmwt/utt.ctm' @ `date`"
fi
