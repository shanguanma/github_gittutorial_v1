#!/bin/bash

. path.sh
. cmd.sh

set -e

echo
echo "## LOG: $0 $@"
echo


# common option
#cmd="slurm.pl --quiet --nodelist=node07"
cmd="slurm.pl  --quiet --exclude=node03,node04,node05,node06,node07"
steps=
#nj=30
nj=100
exp_root=exp

function Example {
 cat<<EOF

 [Example]:sbatch -o log/step1-15.log  $0 --steps 1  --cmd "$cmd" --nj $nj  

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

if [ ! -z $step1 ];then
   #make mfcc, vad, cmvn feature for speaker diarization
   for name in dev_sge_10; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 2 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir

    sid/compute_vad_decision.sh --nj 2 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  
    local/nnet3/xvector/prepare_feats.sh --nj 2 --cmd "$train_cmd" \
      data/$name data/${name}_cmn exp/${name}_cmn
    cp data/$name/vad.scp data/${name}_cmn/
    if [ -f data/$name/segments ]; then
      cp data/$name/segments data/${name}_cmn/
    fi
    utils/fix_data_dir.sh data/${name}_cmn
   done  
 
fi
