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
# I now use other dictionary(it is pure mandarin syllable(e.g:it only contains our team part mandarain name)) to add the old dictionary.   
# the mandarin syllable dictionary is from  mandarin_dict_from_haihua/specified_mandarin_syl.txt                                                                                                
# how to get then mandarin syllable dictionary?
# We just need to use "grep" to select the corresponding name from this file(e.g:mandarin_dict_from_haihua/if-syl-dict-update_v1_remove_tone_v1.txt).


# common option                                                                                       
                                                              
steps=                                                                                                
nj=18                                                                                                 
#nj=50                                                                                                
exp_root=exp                                                                                          
nnet3_affix=_1a
version=v3                                                                                       
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
   
   mandarin_dict_without_tone=mandarin_dict_from_haihua/specified_mandarin_syl.txt
   
   
   new_dict_dir=english_dict_with_specified_mandarin_syllable
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
  dict=english_dict_with_specified_mandarin_syllable
  lang=data/local/english_dict_with_specified_mandarin_syllable_lang
  utils/validate_dict_dir.pl $dict || { echo "## ERROR (step03): failed to validating dict '$dict'" && exit 1;  }
  utils/prepare_lang.sh $dict "<unk>" $lang/tmp $lang
 
fi


lmdir=data/local/lm_english_dict_with_specified_mandarin_syllable
train_data=data/zhiping_eng_text/text-en
dictdir=english_dict_with_specified_mandarin_syllable
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
lang=data/local/english_dict_with_specified_mandarin_syllable_lang
lang_test=data/local/english_dict_with_specified_mandarin_syllable_lang_test
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
graph_dir=$dir/graph_english_dict_with_specified_mandarin_syllable
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
   graph_dir=$dir/graph_english_dict_with_specified_mandarin_syllable
   for decode_set in $test_set; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$train_cmd"  \
          --online-ivector-dir exp/nnet3_1a/ivectors_${decode_set}_${version}_zhiping \
          $graph_dir data/${decode_set} \
          $dir/decode_${decode_set} || exit 1;
  done

# result:asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/scoring_kaldi/best_wer
# WER:%WER 30.97 [ 96 / 310, 6 ins, 13 del, 77 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/wer_10_1.0

fi

# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step09 ];then
   # some check work
   test_data=test_wavdata_$version
   data=data/$test_data
   # here $lang function  only offers words.txt
   #lang=asr_models_english_zhiping/lang-4glm-only-english
   lang=data/local/english_dict_with_specified_mandarin_syllable_lang
   srcdir=asr_models_english_zhiping/chain1024tdnnf
   dir=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_$version
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
# asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/lattice_oracle/log/oracle_overall_wer.log
# result:13.87%
fi

# I used tung's script to modify lattice with hot-list to improve wer
# first experiment: I use his method to modify one pass results(it is lattice type )(one pass result means: it don't use any rescoring method, it only is decode lattices) to improve wer
# second experiment:I try to  use his method to modify oracle lattice to improve wer.
# first experiment:step10-15.
# refrence:/home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/modifyLattice_biasKeywords.sh

# current folder path:/home4/md510/w2019a/kaldi-recipe/chng_project_2019
# Generating key word index lattice requires the follows files as input files. 
lang_dir=data/local/english_dict_with_mandarin_syllable_lang_test # the lang contains G.fst 
model=asr_models_english_zhiping/chain1024tdnnf/final.mdl # acoustic model
# the decode dir is one pass decode results.one pass means that it does not make any rescores.
decodedir=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3
datadir=data/test_wavdata_v3 # test data dir ,it is 40 dimension data folder.
refText=data/test_wavdata_v3/text
# Generating key word index lattice requires the follows configur file
#acwt=0.076923077 # i.e. 1/13: acoustic scale 
#lmwt=1.0 #LM scale
acwt=0.1 # i.e. 1/12: acoustic scale 
lmwt=1.0 #LM scale
wip=1.0


keyword_method_root_dir=exp/keyword_tung_v3
indices=$keyword_method_root_dir/index_data/dev
wbound=${lang_dir}/phones/word_boundary.int

kwsdatadir=$keyword_method_root_dir/datadir
[ -d $kwsdatadir ] || mkdir -p $kwsdatadir
[ -d $indices ] || mkdir -p $indices
if [ ! -z $step10 ];then
  # in further I need to offer segments of train data set.
  # now i use tung's offer. the work to do by myself.
  cp -r /home2/tungpham/hotlist_rescoring_Maduo_v3/datadir_dev_sge/segments $datadir/ 
  cp -r $datadir/segments $kwsdatadir/segments
  utter_id=$kwsdatadir/utter_id
  utter_map=$kwsdatadir/utter_map 
  awk '{print $1,NR}' $kwsdatadir/segments > $utter_id
  awk '{print $1,$2}' $kwsdatadir/segments >$utter_map

  log=$indices/log
  [ -d $log ] ||mkdir -p $log
  nj=`cat $decodedir/num_jobs`
   
  $train_cmd JOB=1:$nj $indices/log/index.JOB.log \
  lattice-align-words $wbound  $model "ark:gzip -cdf $decodedir/lat.JOB.gz|" ark:- \| \
  lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
  lattice-to-kws-index ark:$utter_id ark:- ark:- \| \
  kws-index-union --skip-optimization=true ark:- "ark:|gzip -c > $indices/index.JOB.gz"

  echo "make key word index lattice"
fi

# make keywords fst.
# example is as /home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/keywords.txt
# make it by yourself. These are the so-called keywords or hot words you define.
keywordList=/home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/keywords.txt

if [ ! -z $step11 ];then
  cp -r $decodedir/num_jobs $indices/num_jobs
  nj=`cat $indices/num_jobs`

  #Create FST keywords. Only IV keywords can be converted to fst 
  #(in $kwsdatadir/keywords.fsts), OOV keywords are in $kwsdatadir/oov_kwid.txt
  keyword_text=$kwsdatadir/keywords.txt # input files
  keyword_int=$kwsdatadir/keywords.int  # output files
  keyword_fsts=$kwsdatadir/keywords.fsts # output files
  cp $keywordList $keyword_text
  perl /home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/make_int_kwlist.pl \
       ${lang_dir}/words.txt <$keyword_text >$keyword_int 2> $kwsdatadir/oov_kwid.txt
  echo -e "\n OOV keyword list in $kwsdatadir/oov_kwid.txt \n";
  transcripts-to-fsts ark:$keyword_int ark:$keyword_fsts

fi
kwsoutput=$keyword_method_root_dir/search_results/dev
mkdir -p $kwsoutput
### search 
if [ ! -z $step12 ];then
  
  #steps/search_index.sh \
  # md note:here /home2/tungpham/hotlist_rescoring_Maduo_v3/steps/search_index.sh is different from steps/search_index.sh of offical kaldi
  # 
  /home2/tungpham/hotlist_rescoring_Maduo_v3/steps/search_index.sh \
   --cmd "$train_cmd" --indices-dir $indices \
    $kwsdatadir $kwsoutput keywords.fsts

fi

alignLat=$keyword_method_root_dir/new_lats/alignLat
orgLatMod=$keyword_method_root_dir/new_lats/orgLatMod
alignLatMod=$keyword_method_root_dir/new_lats/alignLatMod
tmp=$keyword_method_root_dir/tmp
mkdir -p $alignLat
mkdir -p $orgLatMod
mkdir -p $alignLatMod
mkdir -p $tmp
nj=`cat $indices/num_jobs`
if [ ! -z $step13 ];then
  for (( i = 1 ; i <= $nj ; i++ ))
  do

  ##Modify lattice
  /home3/md510/kaldi/src/latbin/kaldi_lattice_invest_ver_withTime --acoustic-scale=$acwt $model "ark:gzip -cdf $decodedir/lat.$i.gz|" \
   $wbound $kwsoutput/result.$i $kwsdatadir/keywords.int $tmp/lat_trav.$i $tmp/lat_trav_log.$i.txt \
     "ark:|gzip -c >$alignLat/lat.$i.gz" "ark:|gzip -c >$orgLatMod/lat.$i.gz" "ark:|gzip -c >$alignLatMod/lat.$i.gz" > log.$i.txt
  done
fi

if [ ! -z $step14 ];then
  for (( i=1 ; i <= $nj; i++))
  do
  ##1-best from original lattice
  lattice-1best --lm-scale=1 --acoustic-scale=$acwt --word-ins-penalty=$wip "ark:gzip -cdf $decodedir/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids.$i.txt ark,t:- ark,t:$tmp/lm-costs.$i.txt ark,t:$tmp/acoustic-costs.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > OrgTranscript_${i}_best.hyp

  ##Following two commands are not very important, I just want to test some thing
  lattice-1best --lm-scale=1 --acoustic-scale=$acwt --word-ins-penalty=$wip "ark:gzip -cdf $orgLatMod/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_mod.$i.txt ark,t:- ark,t:$tmp/lm-costs_mod.$i.txt \
   ark,t:$tmp/acoustic-costs_mod.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $orgLatMod/ModTranscript_${i}_best.hyp

  lattice-1best --lm-scale=1 --acoustic-scale=$acwt --word-ins-penalty=$wip "ark:gzip -cdf $alignLat/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_align.$i.txt ark,t:- ark,t:$tmp/lm-costs_align.$i.txt \
   ark,t:$tmp/acoustic-costs_align.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $alignLat/AlignedLattice_${i}_best.hyp

  #This important. It estimate the 1-best from the modified lattice
  lattice-1best --lm-scale=1 --acoustic-scale=1 --word-ins-penalty=$wip "ark:gzip -cdf $alignLatMod/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_alignMod.$i.txt ark,t:- ark,t:$tmp/lm-costs_alignMod.$i.txt \
   ark,t:$tmp/acoustic-costs_alignMod.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $alignLatMod/AlignedLatticeMod_${i}_best.hyp

  done

fi

if [ ! -z $step15 ];then
  ##1-best from original lattice as well as modified lattice
  cat OrgTranscript_*_best.hyp > OrgTranscription_1best.txt
  cat $alignLatMod/AlignedLatticeMod_*_best.hyp > $alignLatMod/AlignedLatticeMod_1best.txt
  compute-wer --text --mode=present ark:$refText  ark:OrgTranscription_1best.txt > wer_orinal_0.0 || exit 1;
  compute-wer --text --mode=present ark:$refText  ark:$alignLatMod/AlignedLatticeMod_1best.txt > $alignLatMod/wer_0.0 || exit 1;
 
fi

# second experiment: :I try to  use his method to modify oracle lattice to improve wer.
# so I only change decodedir,now decodedir is oracle lattice path,the rest file (e.g:lang, am model,test data(data/test_wavdata_v3)) is no changed
# now I set total output root dir , This is convenient for management
keyword_method_root_dir=exp/keyword_tung_v4 # output root dir
decodedir=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/lattice_oracle  # oracle lattice path

# compared first experiment ,the following files are  unchanged.
lang_dir=data/local/english_dict_with_mandarin_syllable_lang_test # the lang contains G.fst 
model=asr_models_english_zhiping/chain1024tdnnf/final.mdl # acoustic model
datadir=data/test_wavdata_v3 # test data dir ,it is 40 dimension data folder.
refText=data/test_wavdata_v3/text
# Generating key word index lattice requires the follows configur file
#acwt=0.076923077 # i.e. 1/13: acoustic scale 
#lmwt=1.0 #LM scale
acwt=0.1 # i.e. 1/12: acoustic scale 
lmwt=1.0 #LM scale
wip=1.0
indices=$keyword_method_root_dir/index_data/dev
wbound=${lang_dir}/phones/word_boundary.int

kwsdatadir=$keyword_method_root_dir/datadir
[ -d $kwsdatadir ] || mkdir -p $kwsdatadir
[ -d $indices ] || mkdir -p $indices

if [ ! -z $step16 ];then
  # in further I need to offer segments of train data set.
  # now i use tung's offer. the work to do by myself.
  cp -r /home2/tungpham/hotlist_rescoring_Maduo_v3/datadir_dev_sge/segments $datadir/
  cp -r $datadir/segments $kwsdatadir/segments
  utter_id=$kwsdatadir/utter_id
  utter_map=$kwsdatadir/utter_map
  awk '{print $1,NR}' $kwsdatadir/segments > $utter_id
  awk '{print $1,$2}' $kwsdatadir/segments >$utter_map

  nj=`cat $decodedir/num_jobs`

  $train_cmd JOB=1:$nj $indices/log/index.JOB.log \
  lattice-align-words $wbound  $model "ark:gzip -cdf $decodedir/lat.JOB.gz|" ark:- \| \
  lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
  lattice-to-kws-index ark:$utter_id ark:- ark:- \| \
  kws-index-union --skip-optimization=true ark:- "ark:|gzip -c > $indices/index.JOB.gz"

  echo "make key word index lattice"
fi

# make keywords fst.
# example is as /home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/keywords.txt
# make it by yourself. These are the so-called keywords or hot words you define.
keywordList=/home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/keywords.txt

if [ ! -z $step17 ];then
  cp -r $decodedir/num_jobs $indices/num_jobs
  nj=`cat $indices/num_jobs`

  #Create FST keywords. Only IV keywords can be converted to fst 
  #(in $kwsdatadir/keywords.fsts), OOV keywords are in $kwsdatadir/oov_kwid.txt
  keyword_text=$kwsdatadir/keywords.txt # input files
  keyword_int=$kwsdatadir/keywords.int  # output files
  keyword_fsts=$kwsdatadir/keywords.fsts # output files
  cp $keywordList $keyword_text
  perl /home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/make_int_kwlist.pl \
       ${lang_dir}/words.txt <$keyword_text >$keyword_int 2> $kwsdatadir/oov_kwid.txt
  echo -e "\n OOV keyword list in $kwsdatadir/oov_kwid.txt \n";
  transcripts-to-fsts ark:$keyword_int ark:$keyword_fsts

fi
kwsoutput=$keyword_method_root_dir/search_results/dev
mkdir -p $kwsoutput
### search 
if [ ! -z $step18 ];then

  #steps/search_index.sh \
  # md note:here /home2/tungpham/hotlist_rescoring_Maduo_v3/steps/search_index.sh is different from steps/search_index.sh of offical kaldi
  # 
  /home2/tungpham/hotlist_rescoring_Maduo_v3/steps/search_index.sh \
   --cmd "$train_cmd" --indices-dir $indices \
    $kwsdatadir $kwsoutput keywords.fsts

fi

alignLat=$keyword_method_root_dir/new_lats/alignLat
orgLatMod=$keyword_method_root_dir/new_lats/orgLatMod
alignLatMod=$keyword_method_root_dir/new_lats/alignLatMod
tmp=$keyword_method_root_dir/tmp
mkdir -p $alignLat
mkdir -p $orgLatMod
mkdir -p $alignLatMod
mkdir -p $tmp
nj=`cat $indices/num_jobs`
if [ ! -z $step19 ];then
  for (( i = 1 ; i <= $nj ; i++ ))
  do

  ##Modify lattice
  /home3/md510/kaldi/src/latbin/kaldi_lattice_invest_ver_withTime --acoustic-scale=$acwt $model "ark:gzip -cdf $decodedir/lat.$i.gz|" \
   $wbound $kwsoutput/result.$i $kwsdatadir/keywords.int $tmp/lat_trav.$i $tmp/lat_trav_log.$i.txt \
     "ark:|gzip -c >$alignLat/lat.$i.gz" "ark:|gzip -c >$orgLatMod/lat.$i.gz" "ark:|gzip -c >$alignLatMod/lat.$i.gz" > $alignLatMod/log.$i.txt
  done
fi

original_lattice_transcript=$keyword_method_root_dir/original_transcript
mkdir -p $original_lattice_transcript
if [ ! -z $step20 ];then
  
  for (( i=1 ; i <= $nj; i++))
  do
  ##1-best from original lattice
  
  lattice-1best --lm-scale=1 --acoustic-scale=$acwt --word-ins-penalty=$wip "ark:gzip -cdf $decodedir/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids.$i.txt ark,t:- ark,t:$tmp/lm-costs.$i.txt ark,t:$tmp/acoustic-costs.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $original_lattice_transcript/OrgTranscript_${i}_best.hyp

  ##Following two commands are not very important, I just want to test some thing
  lattice-1best --lm-scale=1 --acoustic-scale=$acwt --word-ins-penalty=$wip "ark:gzip -cdf $orgLatMod/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_mod.$i.txt ark,t:- ark,t:$tmp/lm-costs_mod.$i.txt \
   ark,t:$tmp/acoustic-costs_mod.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $orgLatMod/ModTranscript_${i}_best.hyp

  lattice-1best --lm-scale=1 --acoustic-scale=$acwt --word-ins-penalty=$wip "ark:gzip -cdf $alignLat/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_align.$i.txt ark,t:- ark,t:$tmp/lm-costs_align.$i.txt \
   ark,t:$tmp/acoustic-costs_align.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $alignLat/AlignedLattice_${i}_best.hyp

  #This important. It estimate the 1-best from the modified lattice
  lattice-1best --lm-scale=1 --acoustic-scale=1 --word-ins-penalty=$wip "ark:gzip -cdf $alignLatMod/lat.$i.gz|" \
   ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_alignMod.$i.txt ark,t:- ark,t:$tmp/lm-costs_alignMod.$i.txt \
   ark,t:$tmp/acoustic-costs_alignMod.$i.txt | utils/int2sym.pl -f 2- $lang_dir/words.txt > $alignLatMod/AlignedLatticeMod_${i}_best.hyp

  done

fi

if [ ! -z $step21 ];then
  ##1-best from original lattice as well as modified lattice
  cat $original_lattice_transcript/OrgTranscript_*_best.hyp > $original_lattice_transcript/OrgTranscription_1best.txt
  cat $alignLatMod/AlignedLatticeMod_*_best.hyp > $alignLatMod/AlignedLatticeMod_1best.txt
  compute-wer --text --mode=present ark:$refText  ark:$original_lattice_transcript/OrgTranscription_1best.txt > $original_lattice_transcript/wer_orinal_0.0 || exit 1;
  compute-wer --text --mode=present ark:$refText  ark:$alignLatMod/AlignedLatticeMod_1best.txt > $alignLatMod/wer_0.0 || exit 1;

fi

if [ ! -z $step22 ];then
   # get word boundary of one pass(top1best lattice) lattice
   # refrence:/home3/md510/kaldi/egs/hub4_english/s5/local/score_sclite.sh
   lmwt=10  # from asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/scoring_kaldi/best_wer
   #word_ins_penalty=0.0,0.5,1.0
   word_ins_penalty=1.0
   factor=$(cat asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/frame_subsampling_factor) || exit 1
   frame_shift_opt="--frame-shift=0.0$factor"
   data=data/test_wavdata_v3
   name=`basename $data`; # e.g.test_wavdata_v3
 
   lang=data/local/english_dict_with_mandarin_syllable_lang # it is a lang
   model=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/chain1024tdnnf/final.mdl
   dir=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3
    for wip in $(echo $word_ins_penalty | sed 's/,/ /g'); do
    $train_cmd LMWT=$lmwt $dir/scoring/log/get_ctm.LMWT.${wip}.log \
      mkdir -p $dir/score_ctm/score_LMWT_${wip}/ '&&' \
      lattice-scale --lm-scale=LMWT "ark:gunzip -c $dir/lat.*.gz|" ark:- \| \
      lattice-add-penalty --word-ins-penalty=$wip ark:- ark:- \| \
      lattice-1best ark:- ark:- \| \
      lattice-align-words $lang/phones/word_boundary.int $model ark:- ark:- \| \
      nbest-to-ctm $frame_shift_opt ark:- - \| \
      utils/int2sym.pl -f 5 $lang/words.txt '>' \
      $dir/score_ctm/score_LMWT_${wip}/$name.utt_ctm || exit 1;
   done
fi

if [ ! -z $step23 ];then
   cat exp/keyword_tung_v3/search_results/dev/result.* | sort -t ' ' -k 2 -d | awk '{print $1,$2,$3,$4 }'>exp/keyword_tung_v3/search_results/dev/result_total
   # exp/keyword_tung_v3/search_results/dev/result_total format is as follows:
   # note:1frame=10ms
   # <keyword-id> <utt-id> <start-frame> <end-frame> 
   # beacause i used tdnnf model, it used 3 subsample frame , in other words ,its frame_subsampling_factor is 3
   # so I  covert frame type to time type for exp/keyword_tung_v3/search_results/dev/result_total 
   awk '{print $1,$2,(3*$3/100),(3*$4/100)}' exp/keyword_tung_v3/search_results/dev/result_total > exp/keyword_tung_v3/search_results/dev/result_total_time_type
   # exp/keyword_tung_v3/search_results/dev/result_total_time_type foramt is as follows:
   # <keyword-id> <utt-id> <start-time> <end-time> 

fi
if [ ! -z $step24 ];then
   keywords_boundary_file=exp/keyword_tung_v3/search_results/dev/result_total_time_type 
   keywords_list=/home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/keywords.txt
   keywords_boundary_align=exp/keyword_tung_v3/search_results/dev/keywords_align
   source-md/egs/modify-decode-lattice/make_replace_keywords_in_search_result.py $keywords_boundary_file  $keywords_list > $keywords_boundary_align

fi
if [ ! -z $step25 ];then
  one_pass_words_boundary_file=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/score_ctm/score_10_1.0/test_wavdata_v3.utt_ctm 
  keywords_boundary_align=exp/keyword_tung_v3/search_results/dev/keywords_align
  modified_one_pass_transcript=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/words_boundary_modified_one_pass/modified_one_pass_transcript
  source-md/egs/modify-decode-lattice/make_replace_keywords_in_ctm.py  $one_pass_words_boundary_file $keywords_boundary_align  $modified_one_pass_transcript
fi

# computer wer after modified one_pass transcript using word boundary information.
if [ ! -z $step26 ];then
    ref=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/scoring_kaldi/test_filt.txt
    modified_one_pass_transcript=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/words_boundary_modified_one_pass/modified_one_pass_transcript
    modified_one_pass_folder=asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/words_boundary_modified_one_pass
    compute-wer --text --mode=present ark:$ref  ark:$modified_one_pass_transcript > $modified_one_pass_folder/wer_0.0 || exit 1;
    # old result(raw one pass):
    # cat asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/scoring_kaldi/best_wer
    #%WER 30.97 [ 96 / 310, 6 ins, 13 del, 77 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/wer_10_1.0 

    # new result:
    #  cat asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/words_boundary_modified_one_pass/wer_0.0 
    #%WER 28.71 [ 89 / 310, 12 ins, 11 del, 66 sub ]
    #%SER 87.50 [ 7 / 8 ]
    #Scored 8 sentences, 0 not present in hyp.

fi

