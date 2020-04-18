#!/bin/bash

. cmd.sh
. path.sh

echo
echo "## LOG $0 $@" 
echo

# current work folder:/home4/md510/w2019a/kaldi-recipe/chng_project_2019                              
# its means the script contains all path is relate the work folder if path isn't absolute Path.       
# The purpose of this script is to use the new dictionary to solve the problem of Chinese name recognition errors.
# the new dictionary contains mandarin syllable,but the dictionary is code-switch dictionary,I now test wave file is pure english ,
# so it doesn't work.                                                                                              
# common option                                                                                       
#cmd="slurm.pl --quiet --nodelist=node07"                                                             
cmd="slurm.pl  --quiet --exclude=node05"                                                              
steps=                                                                                                
nj=18                                                                                                 
#nj=50                                                                                                
exp_root=exp                                                                                          
nnet3_affix=_1a
version=v1                                                                                       
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

# note:  kaldi data format preparation can go to run.sh steps01-02 view.
# full path of run.sh:/home4/md510/w2019a/kaldi-recipe/chng_project_2019/source-md/egs/modify-decode-lattice/run.sh
# here I don't use vad to get subsegment.so I give the data a new name.
# make 40 dimension mfcc features for test set.
if [ ! -z $step01 ];then
  mfccdir=`pwd`/mfcc_$version
  test_data=test_wavdata_$version
  mkdir -p data/$test_data
  cp -r data/test-wavdata-kaldi-format/text data/$test_data/
  cp -r data/test-wavdata-kaldi-format/wav.scp data/$test_data/ 
  cp -r data/test-wavdata-kaldi-format/utt2spk data/$test_data/
  cp -r data/test-wavdata-kaldi-format/spk2utt data/$test_data/

  for dataset in $test_data; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 3 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset} $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset} $exp_root/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}  # remove segments with problems
  done

fi

# Adding mandarin syllable to dictionary to solve the problem,the problem is that the mandarin name is incorrectly identified.
# so I need to update dictionary ,then updating lang, then updating lang_test(it contains G.fst)
# but I use haihua's newest code-switch dictionary(e.g:dictv4.1) . it conatins mandarin syllable
# I use his dictionary to get new lang, then i use zhiping's english text to get lang_test
# dictv4.1 is from /home4/hhx502/w2019/projects/ntu_cts8k_asr_july19/local/dictv4.1 

if [ ! -z $step02 ];then
  dict=dictv4.1
  lang=data/local/dictv4.1_lang
  utils/validate_dict_dir.pl $dict || { echo "## ERROR (step01): failed to validating dict '$dict'" && exit 1;  }
  utils/prepare_lang.sh $dict "<unk>" $lang/tmp $lang
 
fi

if [ ! -z $step03 ];then
  lmdir=data/local/lm_dictv4.1
  train_data=data/zhiping_eng_text/text-en
  # prepared G.fst
  [ -d $lmdir ] || mkdir -p $lmdir
  oov_symbol="<UNK>"
  words_file=data/local/dictv4.1_lang/words.txt

  echo "Using words file: $words_file"
  sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
  cat $train_data > $lmdir/train.txt


 echo "-------------------"
 echo "Maxent 3grams"
 echo "-------------------"
 # sed 's/'${oov_symbol}'/<unk>/g' means: using <unk> to replace ${oov_symbol}
 sed 's/'${oov_symbol}'/<unk>/g' $lmdir/train.txt | \
    ngram-count -lm - -order 3 -text - -vocab $lmdir/vocab -unk -sort -maxent -maxent-convert-to-arpa|\
    sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > $lmdir/3gram.me.gz || exit 1
  echo "## LOG (step03): done with '$lmdir/3gram.me.gz' "
fi

lang_test=data/local/dictv4.1_lang_test
lang=data/local/dictv4.1_lang
lmdir=data/local/lm_dictv4.1
dictdir=dictv4.1
if [ ! -z $step04 ]; then
  utils/format_lm.sh $lang $lmdir/3gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG (step04): done with '$lang_test'"
fi

# make 100 dimension i-vector
# using zhiping's ivector-extractor
if [ ! -z $step05 ];then
  test_data=test_wavdata_$version
  for dataset in $test_data; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 3 \
      data/${dataset} /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/ivector-extractor \
      $exp_root/nnet3${nnet3_affix}/ivectors_${dataset}_${version}_zhiping || exit 1;
  done

fi
dir=asr_models_english_zhiping/chain1024tdnnf
graph_dir=$dir/graph_dictv4.1
if [ ! -z $step06 ];then
   utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $lang_test $dir $graph_dir  || exit 1;
   
fi
# nnet3 decode,
# I used trained chain model(e.g:zhiping's am model) to decode the test dataset.
# zhiping's am model is  /home3/zpz505/w2019/code-switch/update-lexicon-transcript-feb/exp/tdnn/chain1024tdnnf
# because phones.txt of the new lang(e.g:data/local/dictv4.1_lang) mismatch phones.txt of zhiping's am model .
# decoding failed.
if [ ! -z $step07 ];then
   test_set=test_wavdata_$version
   decode_nj=3
   dir=asr_models_english_zhiping/chain1024tdnnf
   graph_dir=$dir/graph_dictv4.1
   for decode_set in $test_set; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "${cmd}"  \
          --online-ivector-dir exp/nnet3_1a/ivectors_${decode_set}_${version}_zhiping \
          $graph_dir data/${decode_set} \
          $dir/decode_${decode_set} || exit 1;
  done

# result:cat asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v1/scoring_kaldi/best_wer
# WER:35.48
fi


