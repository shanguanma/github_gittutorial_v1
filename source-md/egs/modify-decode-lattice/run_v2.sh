#!/bin/bash

. cmd.sh
. path.sh

echo
echo "## LOG $0 $@" 
echo

# current work folder:/home4/md510/w2019a/kaldi-recipe/chng_project_2019                              
# its means the script contains all path is relate the work folder if path isn't absolute Path.      
 
# I now update the pure dictionary by adding mandarin syllable to solve mandarin pinyin name.
# The old dictionary(e.g:asr_models_english_zhiping/dict-only-english) is pure english.
# I now use other dictionary(it is pure mandarin syllable) to add the old dictionary.   
# the mandarin syllable dictionary is from  mandarin_dict_from_haihua/if-syl-dict-update_v1_remove_tone.txt                                                                                                


# common option                                                                                       
                                                              
steps=                                                                                                
nj=18                                                                                                 
#nj=50                                                                                                
exp_root=exp                                                                                          
nnet3_affix=_1a
version=v2                                                                                       
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
# I now update dictionary by add mandarin syllable, to get new dictionary. 
# I use the new dictionary to get new lang, then i use zhiping's english text to get lang_test
#

if [ ! -z $step02 ];then
   # remove pinyin tone
   mandarin_dict_with_tone=mandarin_dict_from_haihua/if-syl-dict-update_v1.txt
   mandarin_dict_without_tone=mandarin_dict_from_haihua/if-syl-dict-update_v1_remove_tone.txt
   source-md/egs/modify-decode-lattice/remove_mandarin_pinyin_tone.py $mandarin_dict_with_tone > $mandarin_dict_without_tone
   
   # --> ERROR: phone "ib5_man" is not in {, non}silence.txt (line 123422)
   # --> ERROR: phone "sil_man" is not in {, non}silence.txt (line 132858)
   # --> ERROR: phone "sil_man" is not in {, non}silence.txt (line 132860)
   # --> ERROR: phone "iong5_man" is not in {, non}silence.txt (line 161059)
   # I don't want to change nonsilence.txt,because it effected am model.
   # I removed them from $mandarin_dict_without_tone ,then get mandarin_dict_from_haihua/if-syl-dict-update_v1_remove_tone_v1.txt 
   new_dict_dir=english_dict_with_mandarin_syllable
   mkdir -p $new_dict_dir
   old_dict_dir=asr_models_english_zhiping/dict-only-english
   cat $old_dict_dir/lexicon.txt mandarin_dict_from_haihua/if-syl-dict-update_v1_remove_tone_v1.txt | sort| uniq >$new_dict_dir/lexicon.txt    
   cp -r $old_dict_dir/extra_questions.txt  $new_dict_dir/
   cp -r $old_dict_dir/nonsilence_phones.txt   $new_dict_dir/ 
   cp -r $old_dict_dir/optional_silence.txt $new_dict_dir/
   cp -r $old_dict_dir/silence_phones.txt $new_dict_dir/ 
   
fi
# make lang
if [ ! -z $step03 ];then
  dict=english_dict_with_mandarin_syllable
  lang=data/local/english_dict_with_mandarin_syllable_lang
  utils/validate_dict_dir.pl $dict || { echo "## ERROR (step03): failed to validating dict '$dict'" && exit 1;  }
  utils/prepare_lang.sh $dict "<unk>" $lang/tmp $lang
 
fi


lmdir=data/local/lm_english_dict_with_mandarin_syllable
train_data=data/zhiping_eng_text/text-en
dictdir=english_dict_with_mandarin_syllable
# prepared G.fst
[ -d $lmdir ] || mkdir -p $lmdir
if [ ! -z $step04 ];then
  cat $train_data | gzip -c > $lmdir/text.gz || exit 1;

  awk '{print $1;}' $dictdir/lexicon.txt \
  |  sort -u | gzip -c > $lmdir/vocab.gz || exit 1;
  #the tool of using is srilm .
  #-text指向输入文件
  #-order指向生成几元的n-gram,即n
  #-lm 指向训练好的语言模型输出文件
  #-vocab ：lm is limited of dict，没有出现在词典的单词将全部替换
  #-unk -sort 去重升序排列
  #-interpolate为插值平滑，-kndiscount为 modified　Kneser-Ney 打折法 ，this two are  used together.
  #this two parameters(-kndiscount -interpolate)are  details at https://blog.csdn.net/xmdxcsj/article/details/50373554
  ngram-count -order 4 -kndiscount -interpolate\
  -vocab $lmdir/vocab.gz  -unk -sort -text $lmdir/text.gz -lm $lmdir/lm4-kn.gz || exit 1;
  echo "## LOG: done with '$lmdir/lm4-kn.gz' "
fi
# make G.fst
lang=data/local/english_dict_with_mandarin_syllable_lang
lang_test=data/local/english_dict_with_mandarin_syllable_lang_test
if [ ! -z $step05 ]; then
  source-md/egs/fisher-english/arpa2G.sh $lmdir/lm4-kn.gz $lang  $lang_test || exit 1;
  echo "## LOG (step06): done with '$lang_test'"
fi


# make 100 dimension i-vector
# using zhiping's ivector-extractor
if [ ! -z $step06 ];then
  test_data=test_wavdata_$version
  for dataset in $test_data; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 3 \
      data/${dataset} /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/ivector-extractor \
      $exp_root/nnet3${nnet3_affix}/ivectors_${dataset}_${version}_zhiping || exit 1;
  done

fi
dir=asr_models_english_zhiping/chain1024tdnnf
graph_dir=$dir/graph_english_dict_with_mandarin_syllable
if [ ! -z $step07 ];then
   utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $lang_test $dir $graph_dir  || exit 1;
   
fi
# nnet3 decode,
# I used trained chain model(e.g:zhiping's am model) to decode the test dataset.
# zhiping's am model is  /home3/zpz505/w2019/code-switch/update-lexicon-transcript-feb/exp/tdnn/chain1024tdnnf
# because phones.txt of the new lang(e.g:data/local/dictv4.1_lang) mismatch phones.txt of zhiping's am model .
# decoding failed.
if [ ! -z $step08 ];then
   test_set=test_wavdata_$version
   decode_nj=3
   dir=asr_models_english_zhiping/chain1024tdnnf
   graph_dir=$dir/graph_english_dict_with_mandarin_syllable
   for decode_set in $test_set; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$train_cmd"  \
          --online-ivector-dir exp/nnet3_1a/ivectors_${decode_set}_${version}_zhiping \
          $graph_dir data/${decode_set} \
          $dir/decode_${decode_set} || exit 1;
  done

# result:cat asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v2/scoring_kaldi/best_wer 
# WER:30.97
fi

# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step09 ];then
   # some check work
   test_data=test_wavdata_$version
   data=data/$test_data
   # here $lang function  only offers words.txt
   #lang=asr_models_english_zhiping/lang-4glm-only-english
   lang=data/local/english_dict_with_mandarin_syllable_lang
   srcdir=asr_models_english_zhiping/chain1024tdnnf
   dir=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v2
   for f in $srcdir/{final.mdl,tree,cmvn_opts} $data/utt2spk $data/feats.scp \
     $lang/words.txt $lang/oov.txt; do
    if [ ! -f $f ]; then
     echo "$0: expected file $f to exist."
     exit 1
    fi
   done

   mkdir -p $dir
   cp $srcdir/final.mdl $dir
   cp $srcdir/tree $dir
   cp $srcdir/cmvn_opts $dir
   cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $dir 2>/dev/null || true
   cp $srcdir/frame_subsampling_factor $dir 2>/dev/null || true

   if [ -f $srcdir/frame_subsampling_factor ]; then
    echo "$0: guessing that this is a chain system, checking parameters."
   fi

  #utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  #cp $lang/phones.txt $dir

  frame_shift_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
  fi
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$train_cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir $dir/lattice_oracle
# result:cat asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v2/lattice_oracle/log/oracle_overall_wer.log
# WER:13.87

fi






