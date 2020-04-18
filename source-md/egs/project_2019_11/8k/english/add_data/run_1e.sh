#!/bin/bash

# 1e used the current best english model to decode the three test set.
# the model is from /home3/zpz505/w2019/release/8k-Oct-AM-Octv4.3-LM-2019 

. path.sh
. cmd.sh
cmd="slurm.pl  --quiet --exclude=node06,node07"
steps=
nj=40
. utils/parse_options.sh || exit 1


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

test_set_1=wiz3ktest
test_set_2=dev_imda_part3_ivr
test_set_3=msf_baby_bonus-8k

tgtdir=run_1e
exp_root=$tgtdir/exp
ivector_extractor=/home3/zpz505/w2019/release/8k-Oct-AM-Octv4.3-LM-2019/ivector-extractor
# get three ivector feature
if [ ! -z $step01 ];then
 for part in $test_set_1 $test_set_2 $test_set_3;do
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 data/${part}_hires \
    $ivector_extractor data/nnet3_1e/${part}_ivectors || exit 1 
 done
fi

if [ ! -z $step02 ];then
 # copy model to my folder
 mkdir -p run_1e/exp
 cp -r /home3/zpz505/w2019/release/8k-Oct-AM-Octv4.3-LM-2019/tdnnf  run_1e/exp/
 graph=run_1e/exp/tdnnf/graph-langv4.3_en_prune
 for part in $test_set_1 $test_set_2 $test_set_3;do 
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
   --nj 10 --cmd "$cmd" \
   --online-ivector-dir data/nnet3_1e/${part}_ivectors \
   $graph data/${part}_hires  run_1e/exp/tdnnf/decode_${part}  || exit 1;
   echo "#$0 LOG Results......"
   source-md/analaysis/analysis_asr_results/show-res.sh  run_1e/exp/tdnnf/decode_${part}
 done
# result :
# cat /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/log/run_1e/steps2.log  
# WER
# wiz3ktext          39.13 
# dev_imda_part3_ivr 30.44
# msf_baby_bonus-8k  27.15

fi

