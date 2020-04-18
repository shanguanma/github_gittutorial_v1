#!/bin/bash

# this script is used to seperate score. 
# seperated score means english score and Chinese score and mix score in english Chinese code-swtch case.

# refrence :espnet/utils/score_sclite.sh
[ -f ./path.sh ] && . ./path.sh

wer=false
bpe=""
bpemodel=""
remove_blank=true
num_spkrs=1
help_message="Usage: $0 <data-dir> <dict>"
steps=
. utils/parse_options.sh

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

if [ $# != 2 ]; then
    echo "${help_message}"
    exit 1;
fi

dir=$1
dict=$2
# 1.concate hyp result to one result.
if [ ! -z $step01 ];then
  concatjson.py ${dir}/data.*.json > ${dir}/data_new.json
fi
# 2. get character ref and  character ref.
if [ ! -z $step02 ];then
  if [ $num_spkrs -eq 1 ]; then
    source-md/egs/asru2019/json2trn_uttid_fisrt_for_release.py ${dir}/data_new.json ${dict} --num-spkrs ${num_spkrs}  --hyps ${dir}/hyp_new.trn
  fi
  if ${remove_blank}; then
      sed -i.bak2 -r 's/<blank> //g' ${dir}/hyp_new.trn
  fi
fi

# 3.get word hyp 
if [ ! -z $step03 ];then
    if [ -n "$bpe" ];then
        spm_decode --model=${bpemodel} --input_format=piece < ${dir}/hyp_new.trn | sed -e "s/▁/ /g" > ${dir}/hyp_new.wrd.trn
    fi
fi

