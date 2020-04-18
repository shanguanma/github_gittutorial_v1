#!/bin/bash

#1b is same as 1a, but I used ntu_inhouse dictionary, this ntu_inhouse dictionary is differnect from

# ditc_ntu_inhouse:/home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse,
# now I will select some word from this biggest ntu dictionary(e.g:/home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs)
# then add to ditc_ntu_inhouse

# how to select word?
# because I add msf data and i2r data to imda data, but for ditc_ntu_inhouse, it has many oov, in order to reduce oov, I will select some oov word 
# from this biggest ntu dictionary.
# then add these selected words to dict_ntu_inhouse.
 

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

tgtdir=run_1b
exp_root=$tgtdir/exp

# make new dictionary
if [ ! -z $step01 ];then
   [ -d $tgtdir/data//ntu_inhouse_add_oov_list ] || mkdir -p $tgtdir/data//ntu_inhouse_add_oov_list 
   #1. get oov word dict from biggest dictionary 
   source-md/egs/project_2019_11/8k/english/add_data/solve_oov_from_big_dict.py data/train_imda_part3_ivr_msf_i2r/from_ntu_inhouse_oov /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs/lexicon.txt > test/oov_dict
   # 2. combine small dictionary
   cat ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  test/oov_dict > run_1b/data/ntu_inhouse_add_oov_list/lexicon.txt
   # 3. show oov by using new dictionary
   cat data/train_imda_part3_ivr_msf_i2r/text |  source-md/egs/acumen/show-oov-with-count.pl --from=2 run_1b/data/ntu_inhouse_add_oov_list/lexicon.txt > data/train_imda_part3_ivr_msf_i2r/from_ntu_inhouse_add_oov_list_oov
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/extra_questions.txt  run_1b/data/ntu_inhouse_add_oov_list/
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/optional_silence.txt run_1b/data/ntu_inhouse_add_oov_list/
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/nonsilence_phones.txt run_1b/data/ntu_inhouse_add_oov_list/ 
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/silence_phones.txt  run_1b/data/ntu_inhouse_add_oov_list/
fi

# make new lang
dictdir=run_1b/data/ntu_inhouse_add_oov_list
lang=run_1b/data/lang
if [ ! -z $step02 ];then
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang


fi

# 3. I directly use tri4(e.g: it is from /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_ntu_inhouse_1a/exp/tri4) 
#  as ali file for these big data(e.g combine MSF_Baby_Bonus_Transcript ,sg-en-i2r  and train_imda_part3_ivr). 
#  so I do not train the gmm-hmm system from scratch 
# 
if [ ! -z $step03 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    data/${train_set}  $lang \
    /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_ntu_inhouse_1a/exp/tri4 \
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
words_file=$tgtdir/data/lang/words.txt
if [ ! -z $step04 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step05 ]; then
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

if [ ! -z $step06 ]; then
  utils/format_lm.sh $tgtdir/data/lang $tgtdir/data/local/lm/4gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi

# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step07 ]; then
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

# because ivector don't change, so I use 1a's ivector feature.
# 5 .train am and decode
if [ ! -z $step08 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1c.sh \
      --stage 1 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang $tgtdir/data/lang \
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


# [md510@node06 add_data]$ cat run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/scoring_kaldi/best_wer 
# %WER 20.53 [ 38091 / 185558, 7346 ins, 8829 del, 21916 sub ] run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/wer_9_0.5
# [md510@node06 add_data]$ cat run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/scoring_kaldi/best_wer 
# %WER 18.18 [ 6326 / 34794, 773 ins, 1423 del, 4130 sub ] run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/wer_9_0.5
# [md510@node06 add_data]$ cat run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
# %WER 38.95 [ 6700 / 17200, 972 ins, 2103 del, 3625 sub ] run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/wer_10_1.0

fi
# without pp 
if [ ! -z $step09 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1c.sh \
      --stage 6 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp false \
      --first-lang $tgtdir/data/lang \
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
# [md510@node06 add_data]$ cat run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better/scoring_kaldi/best_wer 
# %WER 20.58 [ 38194 / 185558, 6885 ins, 9661 del, 21648 sub ] run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better/wer_10_0.0
# [md510@node06 add_data]$ cat run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better/scoring_kaldi/best_wer 
# %WER 19.27 [ 6706 / 34794, 748 ins, 1508 del, 4450 sub ] run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better/wer_10_0.0
# [md510@node06 add_data]$ cat run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better/scoring_kaldi/best_wer 
# %WER 38.78 [ 6670 / 17200, 1114 ins, 1645 del, 3911 sub ] run_1b/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better/wer_10_0.5

fi
