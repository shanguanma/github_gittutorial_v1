#!/bin/bash

# MA DUO 2020-3-11

# make grapheme lexicon to fit specify train corpus.

# step1: I use big lexicon to  get oov
# step2: convert oov word to grapheme lexicon ,then get oov_lexicon
# step3: merge oov_lexicon and big lexicon to get new_lexicon
# step4: I get wordlist from specify train corpus transcript.
# step5: I use wordlist to select from new_lexicon


# note : Here ,The space between first and second columns of the dictionary is a space, not a table.


# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0

cmd="slurm.pl  --quiet --exclude=node06,node05"
steps=1
nj=20

log "$0 $*"
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

. path.sh
. cmd.sh
big_lexicon=/home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/run_ubs2020_cs_big/data/ubs_dictv0.3
if [ ! -z $step01 ];then
   log "Step 1: using big lexicon to get oov for test_1/." 
   source-md/egs/acumen/check_data_oov.pl  $big_lexicon/lexicon.txt  test_1/   
   log "Step 1 is Successfully finished. [elapsed=${SECONDS}s]"
fi

if [ ! -z $step02 ];then
  log "Step 2: convert oov word to grapheme lexicon ,then get oov_lexicon"
  source-md/prepare_kaldi_data/generate_grapheme_lexicon_v1.py \
      --wordlist_en test_1/oov_count.txt > test_1/oov_lexicon.txt  
  log "Step 2 is Successfully finished. [elapsed=${SECONDS}s]"
fi
merged_lexicon=test_1/ubs_dictv0.3_merge
target_lexicon=test_1/maison2_dict
trainset_text=test_1/text
if [ ! -z $step03 ];then
  log "Step 3: merge oov_lexicon and big lexicon to get new_lexicon"
  
  [ -d $target_lexicon ] && rm -rf $target_lexicon 
  [ -d $merged_lexicon ] && rm -rf $merged_lexicon
  mkdir -p $merged_lexicon
  mkdir -p $target_lexicon
  cp -r $big_lexicon/lexicon.txt $merged_lexicon/
  # note: The space between first and second columns of the dictionary is a space, not a table.
  source-md/prepare_kaldi_data/select_lexicon_from_big_lexicon.py \
    --trainset_en_text test_1/text \
    --big_lexicon $merged_lexicon/lexicon.txt > $target_lexicon/lexicon.txt  

   # silence phones, one per line.
   echo SIL >$target_lexicon/silence_phones.txt
   echo "<sss>" >> $target_lexicon/silence_phones.txt
   echo "<oov>" >> $target_lexicon/silence_phones.txt
   echo SIL > $target_lexicon/optional_silence.txt
   # make empty extra_questions.txt, it is ok for make lang.
   touch $target_lexicon/extra_questions.txt
   # get nonsilence_phone.txt from $big_lexicon
   cat $target_lexicon/lexicon.txt | cut -f 2- -d " "  | tr " " "\n" | sort | uniq | grep -E -v "SIL|<sss>|<oov>" >  $target_lexicon/nonsilence_phones.txt
   

fi
if [ ! -z $stepp04 ];then
  log " check lexicon.txt ......"
  num_="$(<  cut -f 2- -d " " $trainset_text | tr " " "\n" | sort | uniq | wc -l )"
  _num="$(< ${target_lexicon}/lexicon.txt  wc -l )"
  if [$num_ ne $_num]; then
      log " make lexicon.txt may be error, because it can't covert all train set text word"
      exit 1;
  else  
      log "make lexicon.txt successfully"
  fi
fi
