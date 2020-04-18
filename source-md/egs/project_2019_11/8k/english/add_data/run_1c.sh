#!/bin/bash

#1c is same as 1b, but I add SWB1 data(e.g:/home4/asr_resource/data/acoustic/english/8k/SWB1/) to train_imda_part3_ivr_msf_i2r 

# ntu_inhouse_add_oov_list: /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/run_1b/data/ntu_inhouse_add_oov_list
# it is a dictionary.
# now I will select some word from this biggest ntu dictionary(e.g:/home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs)
# then add to ntu_inhouse_add_oov_list

# how to select word?
# because I add SWB1 data, but for ntu_inhouse_add_oov_list, it has many oov, in order to reduce oov, I will select some oov word 
# from this biggest ntu dictionary.
# then add these selected words to ntu_inhouse_add_oov_list.


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

train_set=train_imda_part3_ivr_msf_i2r_SWB1

test_set_1=wiz3ktest
test_set_2=dev_imda_part3_ivr
test_set_3=msf_baby_bonus-8k

tgtdir=run_1c
exp_root=$tgtdir/exp
add_data_1=SWB1
# make mfcc for SWB1 data
feat_dir=feat/mfcc
if [ ! -z $step01 ];then
   cp -r /home4/asr_resource/data/acoustic/english/8k/SWB1 data/ 
   source-md/egs/project_2019_11/8k/english/add_data/downsample_sph_file_for_SWB1.py  data/SWB1/wav.scp > data/SWB1/wav_1.scp
   head -n 5 data/SWB1/wav_1.scp
   mv data/SWB1/wav_1.scp data/SWB1/wav.scp   
   for part in SWB1; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
  done

fi
# combine data
if [ ! -z $step02 ];then
    # get recod2dur file ,in order to get speed perturbed data
    utils/data/get_reco2dur.sh data/SWB1 
    utils/combine_data.sh  data/$train_set data/train_imda_part3_ivr_msf_i2r  data/SWB1
fi
# make new dictionary
if [ ! -z $step03 ];then
   [ -d $tgtdir/data/ntu_inhouse_add_oov_list_SWB1 ] || mkdir -p $tgtdir/data/ntu_inhouse_add_oov_list_SWB1
   # 0. get oov
   cat data/train_imda_part3_ivr_msf_i2r_SWB1/text | source-md/egs/acumen/show-oov-with-count.pl --from=2  ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  > data/train_imda_part3_ivr_msf_i2r_SWB1/from_ntu_inhouse_oov
   #1. get oov word dict from biggest dictionary 
   source-md/egs/project_2019_11/8k/english/add_data/solve_oov_from_big_dict.py \
     data/train_imda_part3_ivr_msf_i2r_SWB1/from_ntu_inhouse_oov \
     /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs/lexicon.txt > test/oov_dict
   # 2. combine small dictionary
   cat ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  test/oov_dict | uniq > $tgtdir/data/ntu_inhouse_add_oov_list_SWB1/lexicon.txt
   # 3. show oov by using new dictionary
   cat data/train_imda_part3_ivr_msf_i2r_SWB1/text |  source-md/egs/acumen/show-oov-with-count.pl --from=2 run_1c/data/ntu_inhouse_add_oov_list_SWB1/lexicon.txt > data/train_imda_part3_ivr_msf_i2r_SWB1/from_ntu_inhouse_add_oov_list_SWB1_oov
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/extra_questions.txt  $tgtdir/data/ntu_inhouse_add_oov_list_SWB1
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/optional_silence.txt $tgtdir/data/ntu_inhouse_add_oov_list_SWB1
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/nonsilence_phones.txt $tgtdir/data/ntu_inhouse_add_oov_list_SWB1
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/silence_phones.txt  $tgtdir/data/ntu_inhouse_add_oov_list_SWB1
fi

# make new lang
dictdir=$tgtdir/data/ntu_inhouse_add_oov_list_SWB1
lang=$tgtdir/data/lang
if [ ! -z $step04 ];then
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang


fi

# 3. I directly use tri4(e.g: it is from /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_ntu_inhouse_1a/exp/tri4) 
#  as ali file for these big data(e.g combine MSF_Baby_Bonus_Transcript ,sg-en-i2r  and train_imda_part3_ivr). 
#  so I do not train the gmm-hmm system from scratch 
if [ ! -z $step05 ]; then
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
if [ ! -z $step06 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step07 ]; then
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

if [ ! -z $step08 ]; then
  utils/format_lm.sh $tgtdir/data/lang $tgtdir/data/local/lm/4gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi

# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step09 ]; then
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

# 4 .prepare ivector for these big train data(e.g combine MSF_Baby_Bonus_Transcript ,sg-en-i2r,  SWB1 and train_imda_part3_ivr ).
#    prepare ivector for three test set
if [ ! -z $step10 ];then
    source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/nnet3/run_ivector_8k_1d.sh \
       --stage 10 \
       --speed-perturb true \
       --train-set $train_set \
       --test-set-1  $test_set_1 \
       --test-set-2  $test_set_2 \
       --test-set-3  $test_set_3 \
       --nnet3-affix _1a \
       --test-set-is-run-mfcc-hires false

fi
#.
# 5 .train am and decode
if [ ! -z $step11 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1c.sh \
      --stage 6 \
      --train-stage 0 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir data/train_imda_part3_ivr_msf_i2r_SWB1 \
      --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_SWB1_sp \
      --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_SWB1_sp_hires \
      --train-ivector-dir  data/nnet3_1a/ivectors_train_imda_part3_ivr_msf_i2r_SWB1_sp_hires \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3_1a

# result 
# %WER 21.12 [ 39193 / 185558, 6416 ins, 10694 del, 22083 sub ] run_1c/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/wer_9_1.0
#[md510@node06 add_data]$ cat run_1c/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/scoring_kaldi/best_wer 
#%WER 24.42 [ 8496 / 34794, 948 ins, 2586 del, 4962 sub ] run_1c/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/wer_10_0.0
#[md510@node06 add_data]$ cat run_1c/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
#%WER 39.76 [ 6838 / 17200, 1268 ins, 1617 del, 3953 sub ] run_1c/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/wer_10_0.5
fi
