#!/bin/bash
# 1f is same as 1b, but it doesn't use speed perturbed on ivector.


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

tgtdir=run_1f
exp_root=$tgtdir/exp

# make new dictionary
if [ ! -z $step01 ];then
   [ -d $tgtdir/data/ntu_inhouse_add_oov_list ] || mkdir -p $tgtdir/data/ntu_inhouse_add_oov_list
   #0. get oov word
   cat data/train_imda_part3_ivr_msf_i2r/text |  source-md/egs/acumen/show-oov-with-count.pl --from=2 ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt >data/train_imda_part3_ivr_msf_i2r/from_ntu_inhouse_oov
   #1. get oov word dict from biggest dictionary 
   source-md/egs/project_2019_11/8k/english/add_data/solve_oov_from_big_dict.py data/train_imda_part3_ivr_msf_i2r/from_ntu_inhouse_oov /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs/lexicon.txt > test/oov_dict
   # 2. combine small dictionary
   cat ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  test/oov_dict > $tgtdir/data/ntu_inhouse_add_oov_list/lexicon.txt
   # 3. show oov by using new dictionary
   cat data/train_imda_part3_ivr_msf_i2r/text |  source-md/egs/acumen/show-oov-with-count.pl --from=2 $tgtdir/data/ntu_inhouse_add_oov_list/lexicon.txt > data/train_imda_part3_ivr_msf_i2r/from_ntu_inhouse_add_oov_list_oov
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/extra_questions.txt  $tgtdir/data/ntu_inhouse_add_oov_list/
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/optional_silence.txt $tgtdir/data/ntu_inhouse_add_oov_list/
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/nonsilence_phones.txt $tgtdir/data/ntu_inhouse_add_oov_list/
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/silence_phones.txt  $tgtdir/data/ntu_inhouse_add_oov_list/
fi

# make new lang
dictdir=$tgtdir/data/ntu_inhouse_add_oov_list
lang=$tgtdir/data/lang
if [ ! -z $step02 ];then
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang

fi

###############################################################################
# pepared new lang_test 
# more detail ,you can read /home4/md510/w2019a/kaldi-recipe/lm/data/lm/perplexities.txt
#                           /home4/md510/w2019a/kaldi-recipe/lm/data/lm/ 
#                           /home4/md510/w2019a/kaldi-recipe/lm/train_lms_srilm.sh 
###############################################################################
datadir=data
lmdir=$tgtdir/data/local/lm
train_data=$datadir/$train_set
# prepared G.fst
[ -d $lmdir ] || mkdir -p $lmdir
oov_symbol="<UNK>"
words_file=$tgtdir/data/lang/words.txt
if [ ! -z $step03 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step04 ]; then
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

if [ ! -z $step05 ]; then
  utils/format_lm.sh $tgtdir/data/lang $tgtdir/data/local/lm/4gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi


if [ ! -z $step06 ]; then
  # we want to start the
  # monophone training on relatively short utterances (easier to align), but not
  # only the shortest ones (mostly uh-huh).  So take the 100k shortest ones, and
  # then take 30k random utterances from those
  utils/subset_data_dir.sh --shortest  $train_data 100000 $datadir/${train_set}_100kshort
  utils/subset_data_dir.sh $datadir/${train_set}_100kshort 30000 $datadir/${train_set}_30kshort

  # Take the first 200k utterances (just under half the data); we'll use
  # this for later stages of training.
  utils/subset_data_dir.sh --first $train_data  200000 $datadir/${train_set}_200k
  utils/data/remove_dup_utts.sh 200 $datadir/${train_set}_200k $datadir/${train_set}_200k_nodup

  # Finally, the full training set:
  utils/data/remove_dup_utts.sh 300 $train_data $datadir/${train_set}_nodup
fi

###############################################################################
# GMM system training using add augment seame supervised data
###############################################################################
exp_root=$tgtdir/exp
if [ ! -z $step07 ]; then
  steps/train_mono.sh --nj  $nj  --cmd "$cmd" \
    $datadir/${train_set}_30kshort $lang $exp_root/mono || exit 1
fi

if [ ! -z $step08 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set}_200k_nodup $lang $exp_root/mono $exp_root/mono_ali || exit 1

  steps/train_deltas.sh --cmd "$cmd" \
    2500 20000 $datadir/${train_set}_200k_nodup $lang $exp_root/mono_ali $exp_root/tri1 || exit 1

fi

if [ ! -z $step09 ]; then
  # The 100k_nodup data is used in the nnet2 recipe.
  steps/align_si.sh --nj 30 --cmd "$train_cmd" \
                    $datadir/${train_set}_200k_nodup  $lang $exp_root/tri1 $exp_root/tri1_ali_200k_nodup

  # From now, we start using all of the data (except some duplicates of common
  # utterances, which don't really contribute much).
  steps/align_si.sh --nj $nj --cmd "$cmd" \
   $datadir/${train_set}_nodup $lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
    2500 20000 $datadir/${train_set}_nodup $lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

fi

if [ ! -z $step10 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set}_nodup $lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 $datadir/${train_set}_nodup $lang $exp_root/tri2_ali $exp_root/tri3 || exit 1;

fi

if [ ! -z $step11 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set}_nodup  $lang $exp_root/tri3 $exp_root/tri3_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 $datadir/${train_set}_nodup  $lang $exp_root/tri3_ali $exp_root/tri4 || exit 1;

fi

# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step12 ]; then
   # Now we compute the pronunciation and silence probabilities from training data,
   # and re-create the lang directory.
   steps/get_prons.sh --cmd "$cmd" \
    $datadir/${train_set}_nodup $lang $exp_root/tri4

   utils/dict_dir_add_pronprobs.sh --max-normalize true \
    $dictdir \
    $exp_root/tri4/pron_counts_nowb.txt $exp_root/tri4/sil_counts_nowb.txt \
    $exp_root/tri4/pron_bigram_counts_nowb.txt $tgtdir/data/local/dict_pp

   utils/prepare_lang.sh $tgtdir/data/local/dict_pp \
    "<unk>" $tgtdir/data/lang_pp/tmp $tgtdir/data/lang_pp
   utils/format_lm.sh $tgtdir/data/lang_pp $tgtdir/data/local/lm/4gram.me.gz \
     $dictdir/lexiconp.txt $tgtdir/data/lang_test_pp

fi



# train i-vector
if [ ! -z $step13 ];then
    source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/nnet3/run_ivector_8k_1e.sh \
       --stage 1 \
       --speed-perturb false \
       --train-set ${train_set}_nodup \
       --test-set-1  $test_set_1 \
       --test-set-2  $test_set_2 \
       --test-set-3  $test_set_3 \
       --nnet3-affix "_1f"

fi

if [ ! -z $step14 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1d.sh \
      --stage 1 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir data/train_imda_part3_ivr_msf_i2r_nodup \
      --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_nodup \
      --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_nodup_hires_nosp   \
      --train-ivector-dir  data/nnet3_1f/ivectors_train_imda_part3_ivr_msf_i2r_nodup_hires_nosp  \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3_1f
# results
# [md510@node06 add_data]$ cat run_1f/exp/chain/tdnn_cnn_tdnnf/decode_dev_imda_part3_ivr_better_pp/scoring_kaldi/best_wer 
# %WER 20.73 [ 38465 / 185558, 7092 ins, 9722 del, 21651 sub ] run_1f/exp/chain/tdnn_cnn_tdnnf/decode_dev_imda_part3_ivr_better_pp/wer_9_0.5
# [md510@node06 add_data]$ cat run_1f/exp/chain/tdnn_cnn_tdnnf/decode_msf_baby_bonus-8k_better_pp/scoring_kaldi/best_wer 
# %WER 19.45 [ 6766 / 34794, 785 ins, 1850 del, 4131 sub ] run_1f/exp/chain/tdnn_cnn_tdnnf/decode_msf_baby_bonus-8k_better_pp/wer_10_0.0
# [md510@node06 add_data]$ cat run_1f/exp/chain/tdnn_cnn_tdnnf/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
# %WER 39.76 [ 6838 / 17200, 1191 ins, 1804 del, 3843 sub ] run_1f/exp/chain/tdnn_cnn_tdnnf/decode_wiz3ktest_better_pp/wer_10_0.5

fi

# get without pp decode result.
if [ ! -z $step15 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1d.sh \
      --stage 6 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp false \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir data/train_imda_part3_ivr_msf_i2r_nodup \
      --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_nodup \
      --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_nodup_hires_nosp   \
      --train-ivector-dir  data/nnet3_1f/ivectors_train_imda_part3_ivr_msf_i2r_nodup_hires_nosp  \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3_1f
# [md510@node06 add_data]$ cat run_1f/exp/chain/tdnn_cnn_tdnnf/decode_dev_imda_part3_ivr_better/scoring_kaldi/best_wer 
# %WER 20.80 [ 38601 / 185558, 6318 ins, 10704 del, 21579 sub ] run_1f/exp/chain/tdnn_cnn_tdnnf/decode_dev_imda_part3_ivr_better/wer_9_0.5
# [md510@node06 add_data]$ cat run_1f/exp/chain/tdnn_cnn_tdnnf/decode_msf_baby_bonus-8k_better/scoring_kaldi/best_wer 
# %WER 20.41 [ 7101 / 34794, 821 ins, 1683 del, 4597 sub ] run_1f/exp/chain/tdnn_cnn_tdnnf/decode_msf_baby_bonus-8k_better/wer_9_0.0
# [md510@node06 add_data]$ cat run_1f/exp/chain/tdnn_cnn_tdnnf/decode_wiz3ktest_better/scoring_kaldi/best_wer 
# %WER 39.77 [ 6840 / 17200, 1084 ins, 1908 del, 3848 sub ] run_1f/exp/chain/tdnn_cnn_tdnnf/decode_wiz3ktest_better/wer_11_0.0


fi

