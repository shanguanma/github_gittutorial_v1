#!/bin/bash
# Copyright 2019 Ma Duo

# this script want to get phone boundary( it is a ctm file) from a decoding directory that has lattices  present
#                     get word boundary ( it is a ctm file) from a decoding directory that has lattices  present
#                     you can see steps/get_ctm.sh, get_ctm_fast.sh,  get_ctm_conf.sh 
# learning from /home4/md510/w2019a/kaldi_20191103/kaldi/egs/wsj/s5/steps/get_ctm.sh

. path.sh
. cmd.sh
cmd="slurm.pl --quiet"
steps=1
LMWT=12  # it is from /path/to/scoring_kaldi/best_wer
. ./utils/parse_options.sh || exit 1;

echo "$0 $@"  # Print the command line for logging

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

if [ $# -ne 2 ]; then
  echo "Usage: $0 [options] <decode-dir> <lang-dir|graph-dir>"
  echo " Options:"
  echo "    --cmd (run.pl|queue.pl...)      # specify how to run the sub-processes."
  echo "    --steps (1|2)                   # start scoring script from part-way through."
  echo "e.g.:"
  echo "$0 --steps 1 exp/chain/decode/ data/lang"
  echo "See also: steps/get_train_ctm.sh, steps/get_ctm_fast.sh, steps/get_ctm_conf.sh"

  exit 1;
fi

dir=$1 # e.g: asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented   # decoder folder
lang=$2 # e.g: asr_models_english_zhiping/lang-only-english                                          # lang, it hasn't G.fst


nj=$(cat $dir/num_jobs)
lats=$(for n in $(seq $nj); do echo -n "$dir/lat.$n.gz "; done)

# Purpose of step01 or steps02 to get the phone boundary imformation from the decoded lattice.
# Both  steps01 and steps02  are seame, however steps01 doesn't have Intermediate results
#                                              steps02 have result in per substep. 
if [ ! -z $step01 ];then
  $cmd LMWT=12 $dir/scoring_kaldi/phone_boundary.log \
   lattice-1best --lm-scale=LMWT  "ark:gunzip -c $lats|" ark:- \| \
   nbest-to-linear ark:- ark,t:- ark:/dev/null ark:/dev/null \| \
   ali-to-phones --ctm-output $dir/final.mdl ark,t:- - \| \
   utils/int2sym.pl -f 5 $lang/phones.txt '>' $dir/phone_boundary_result_ctm.txt 

fi

if [ ! -z $step02 ];then
   # nbest-to-linear has one input : "ark:gunzip -c $dir/lat.1.gz |"
   #                     one output: $dir/lat_1_1best.lats
   lattice-1best --lm-scale=LMWT  "ark:gunzip -c $dir/lat.1.gz |" ark,t:$dir/lat_1_1best.lats
   # nbest-to-linear has one input : nbest/1best lattice eg:$dir/lat_1_1best.lats
   #                     three output:  alignments, acoustic and LM costs (note: use ark:/dev/null for unwanted outputs)
   nbest-to-linear ark,t:$dir/lat_1_1best.lats ark,t:$dir/lat_1.ali ark:/dev/null ark:/dev/null 
   ali-to-phones --ctm-output --write-lengths=true $dir/final.mdl ark,t:$dir/lat_1.ali - \| \
   utils/int2sym.pl -f $lang/phones.txt >$dir/lat_1.ctm 
fi




