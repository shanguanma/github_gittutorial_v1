#!/bin/bash

# The function of the script add two data(e.g : MSF_Baby_Bonus_Transcript and sg-en-i2r) to train_imda_part3_ivr , then get better result.
# I directly use tri4 of train_imda_part3_ivr as gmm-hmm ali here.

# dictionary : imda_dict 
#             path: /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/dict_imda 


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

train_set=train_imda_part3_ivr_msf_i2r

test_set_1=wiz3ktest
test_set_2=dev_imda_part3_ivr
test_set_3=msf_baby_bonus-8k

tgtdir=run_1a
exp_root=$tgtdir/exp
# 1. prepared data
if [ ! -z $step01 ];then
   # train data set
   [ -d data] || mkdir -p data 
   cp -r /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/train_imda_part3_ivr data/
   cp -r /home4/asr_resource/data/acoustic/english/16k/MSF_Baby_Bonus_Transcript data/
   cp -r /home4/asr_resource/data/acoustic/english/8k/sg-en-i2r data/
   # test data set
     
   cp -r /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/dev_imda_part3_ivr data/
   cp -r /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/wiz3ktest data/
   [ -d data/msf_baby_bonus-8k ] || mkdir -p data/msf_baby_bonus-8k 
   cp -r  /home4/asr_resource/data/dev/msf_baby_bonus-8k/mfcc-hires/*  data/msf_baby_bonus-8k
fi

# 2 . make lang 
dictdir=data/dict_imda
lang=data/lang
if [ ! -z $step02 ];then
  cp -r /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/dict_imda data/ 
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang
 

fi

# 3 . make mfcc for train and test set
feat_dir=feat/mfcc
if [ ! -z $step03 ];then
   for part in train_imda_part3_ivr  dev_imda_part3_ivr wiz3ktest msf_baby_bonus-8k; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
  done

fi

if [ ! -z $step04 ];then
  for part in  MSF_Baby_Bonus_Transcript ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
  done
fi

if [ ! -z $step05 ];then
  for part in sg-en-i2r ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
  done
fi
# 4. combile train set 
if [ ! -z $step06 ];then
   utils/combine_data.sh data/${train_set} data/train_imda_part3_ivr data/MSF_Baby_Bonus_Transcript data/sg-en-i2r

fi

# 3. I directly use tri4(e.g: it is from /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/) 
#  as ali file for these big data(e.g combine MSF_Baby_Bonus_Transcript ,sg-en-i2r  and train_imda_part3_ivr). 
#  so I do not train the gmm-hmm system from scratch 
# 
if [ ! -z $step07 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    data/${train_set}  $lang \
    /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/exp/tri4 \
    $exp_root/tri4_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 data/${train_set}  $lang $exp_root/tri4_ali $exp_root/tri5 || exit 1;

fi


###############################################################################
# pepared new lang_test 
# more detail ,you can read /home4/md510/w2019a/kaldi-recipe/lm/data/lm/perplexities.txt
#                           /home4/md510/w2019a/kaldi-recipe/lm/data/lm/ 
#                           /home4/md510/w2019a/kaldi-recipe/lm/train_lms_srilm.sh 
###############################################################################
lmdir=$tgtdir/data/local/lm
train_data=data/${train_set} 
# prepared G.fst
[ -d $lmdir ] || mkdir -p $lmdir
oov_symbol="<UNK>"
words_file=data/lang/words.txt
if [ ! -z $step08 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step09 ]; then
  echo "-------------------"
  echo "Maxent 4grams"
  echo "-------------------"
  # sed 's/'${oov_symbol}'/<unk>/g' means: using <unk> to replace ${oov_symbol}
  sed 's/'${oov_symbol}'/<unk>/g' $lmdir/train.txt | \
    ngram-count -lm - -order 4 -text - -vocab $lmdir/vocab -unk -sort -maxent -maxent-convert-to-arpa| \
   sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > $lmdir/4gram.me.gz || exit 1
  echo "## LOG : done with '$lmdir/4gram.me.gz'"
fi


lang_test=$tgtdir/data/local/lang_test
[ -d $lang_test ] || mkdir -p $lang_test

if [ ! -z $step10 ]; then
  utils/format_lm.sh data/lang $tgtdir/data/local/lm/4gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi



# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step11 ]; then
   # Now we compute the pronunciation and silence probabilities from training data,
   # and re-create the lang directory.
   steps/get_prons.sh --cmd "$cmd" \
    data/${train_set} $lang $exp_root/tri5

   utils/dict_dir_add_pronprobs.sh --max-normalize true \
    $dictdir \
    $exp_root/tri5/pron_counts_nowb.txt $exp_root/tri5/sil_counts_nowb.txt \
    $exp_root/tri5/pron_bigram_counts_nowb.txt $tgtdir/data/local/dict_pp

   utils/prepare_lang.sh $tgtdir/data/local/dict_pp \
    "<unk>" $tgtdir/data/lang_pp/tmp $tgtdir/data/lang_pp
   utils/format_lm.sh $tgtdir/data/lang_pp $tgtdir/data/local/lm/4gram.me.gz \
     $dictdir/lexiconp.txt $tgtdir/data/lang_test_pp

fi
          

# 4 .prepare ivector for these big train data(e.g combine MSF_Baby_Bonus_Transcript ,sg-en-i2r  and train_imda_part3_ivr).
#    prepare ivector for three test set
if [ ! -z $step12 ];then
    source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/nnet3/run_ivector_8k_1c.sh \
       --stage 1 \
       --speed-perturb true \
       --train-set $train_set \
       --test-set-1  $test_set_1 \
       --test-set-2  $test_set_2 \
       --test-set-3  $test_set_3

fi

# 5 .train am and decode
if [ ! -z $step13 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1c.sh \
      --stage 6 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir data/train_imda_part3_ivr_msf_i2r \
      --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_sp \
      --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_sp_hires \
      --train-ivector-dir  data/nnet3/ivectors_train_imda_part3_ivr_msf_i2r_sp_hires \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3


# reaults:

# [md510@node06 add_data]$ cat run_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/scoring_kaldi/best_wer 
# %WER 20.13 [ 37355 / 185558, 6668 ins, 9744 del, 20943 sub ] run_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/wer_9_0.5
# [md510@node06 add_data]$ cat run_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/scoring_kaldi/best_wer 
# %WER 17.05 [ 5933 / 34794, 754 ins, 1594 del, 3585 sub ] run_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/wer_10_0.0
# [md510@node06 add_data]$ cat run_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
# %WER 39.08 [ 6722 / 17200, 1128 ins, 1950 del, 3644 sub ] run_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/wer_10_0.5
 
fi

