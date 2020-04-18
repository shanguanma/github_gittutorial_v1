#!/bin/bash

# lexicon:
# IMDA Lexicon: /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/local/dict/dict_imda/

# AM Training Data:
# /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/train_imda_part3_ivr{mfcc,mfcc_hires}

# LM Training Data：
# Use training transcripts to build 4-gram language models

#Test sets:
# /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/wiz3ktest/mfcc_hires ,now rawdata path:/home4/asr_resource/data/acoustic/english/8k/wizdata/audio/wizdata/
# /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/dev_imda_part3_ivr


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

# copy data
src_data_dir=/home4/hhx502/w2019/projects/nov03_sge8k/exp/data
train_set=train_imda_part3_ivr
test_set_1=wiz3ktest
test_set_2=dev_imda_part3_ivr
src_dict_dir=/home4/hhx502/w2019/projects/nov03_sge8k/exp/data/local/dict/dict_imda
tgtdir=run_imda_1a

datadir=$tgtdir/data
mkdir -p ${datadir}
if [ ! -z $step01 ]; then
   mkdir -p $datadir/$test_set_1
   mkdir -p $tgtdir/data/$test_set_2
   mkdir -p  $tgtdir/data/$train_set
   # train set copy
   cp -r /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/train_imda_part3_ivr/mfcc/* $datadir/$train_set
   # two test copy
   cp -r  /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/wiz3ktest/mfcc_hires/* $datadir/wiz3ktest
   cp -r /home4/hhx502/w2019/projects/nov03_sge8k/exp/data/dev_imda_part3_ivr/* $datadir/dev_imda_part3_ivr
   # dict copy
   cp -r $src_dict_dir  $datadir
fi

###############################################################################
#prepare lang .
###############################################################################
dictdir=$datadir/dict_imda
lang=$datadir/lang
if [ ! -z $step02 ]; then
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang
fi

###############################################################################
# pepared new lang_test 
# more detail ,you can read /home4/md510/w2019a/kaldi-recipe/lm/data/lm/perplexities.txt
#                           /home4/md510/w2019a/kaldi-recipe/lm/data/lm/ 
#                           /home4/md510/w2019a/kaldi-recipe/lm/train_lms_srilm.sh 
###############################################################################
lmdir=$tgtdir/data/local/lm
train_data=${datadir}/$train_set
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
    source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/nnet3/run_ivector_8k_1b.sh \
       --stage 4 \
       --speed-perturb true \
       --tgtdir $tgtdir \
       --train-set $train_set \
       --test-set-1  $test_set_1 \
       --test-set-2  $test_set_2 

fi

if [ ! -z $step14 ];then
   source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1a.sh \
      --stage 6 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 
# %WER 20.37 [ 37796 / 185558, 6721 ins, 9782 del, 21293 sub ] run_imda_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/wer_9_0.5 # at hangzhou /home/md510/w2019/project/kaldi-recipe/egs/run_imda_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp
# [md510@node05 lexicon_comparison]$ cat run_imda_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
# %WER 54.29 [ 9338 / 17200, 698 ins, 4218 del, 4422 sub ] run_imda_1a/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/wer_9_0.5 
fi
# get 13 mfcc for test set , in order to decode it.
if [ ! -z $step15 ];then
  for sdata in $test_set_1 $test_set_2; do
    # this must be make_mfcc ,it shouldn't  add pitch. otherwise running steps/align_fmllr_lats.sh in step18 local/semisup/chain/run_tdnn.sh is error. 
    # beacuse train_aug folder only contain utt2uniq  spk2utt  text  utt2spk  wav.scp,  so must set -write-utt2num-frames is false 
    # beacuse this doesn't use energy(在conf/mfcc.conf可以看到), so　mfcc features is 13  MFCCs, so mfcc dimension is 13
    # if using energy, so mfcc features is 12 MFCCs + Energy, so mfcc dimension is 13
    mfccdir=$tgtdir/mfcc
    steps/make_mfcc.sh --cmd "$cmd" --nj 10 \
      --mfcc-config conf/mfcc_8k.conf  --write-utt2num-frames false $tgtdir/data/${sdata} $tgtdir/exp/make_mfcc/${sdata} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh $tgtdir/data/${sdata} $tgtdir/exp/make_mfcc/${sdata} $mfccdir || exit 1;
    #This script will fix sorting errors and will remove any utterances for which some required data, such as feature data or transcripts, is missing.
    utils/fix_data_dir.sh $tgtdir/data/${sdata}
    echo "## LOG : done with mfcc feat"

  done
fi

# decode tri4
if [ ! -z $step16 ];then
    utils/mkgraph.sh $lang_test $exp_root/tri4 $exp_root/tri4/graph  || exit 1;
    for decode_set in $test_set_1 ; do 
     steps/decode_fmllr.sh --nj 10 --cmd "$train_cmd" --config conf/decode.config \
      $exp_root/tri4/graph $tgtdir/data/$decode_set $exp_root/tri4/decode_$decode_set  || exit 1;
    done

# [md510@node06 lexicon_comparison]$ cat run_imda_1a/exp/tri4/decode_wiz3ktest/scoring_kaldi/best_wer 
# %WER 85.43 [ 14694 / 17200, 717 ins, 7346 del, 6631 sub ] run_imda_1a/exp/tri4/decode_wiz3ktest/wer_12_1.0
fi

# decode tri3
if [ ! -z $step17 ];then
    utils/mkgraph.sh $lang_test $exp_root/tri3 $exp_root/tri3/graph  || exit 1;
    for decode_set in $test_set_1 $test_set_2; do
      steps/decode.sh  --nj 10 --cmd "$train_cmd" --config conf/decode.config \
      $exp_root/tri3/graph $tgtdir/data/$decode_set $exp_root/tri3/decode_$decode_set  || exit 1;
    done

# [md510@node06 lexicon_comparison]$ cat run_imda_1a/exp/tri3/decode_dev_imda_part3_ivr/scoring_kaldi/best_wer 
# %WER 45.19 [ 83845 / 185558, 12494 ins, 17465 del, 53886 sub ] run_imda_1a/exp/tri3/decode_dev_imda_part3_ivr/wer_15_0.5
# [md510@node06 lexicon_comparison]$ cat run_imda_1a/exp/tri3/decode_wiz3ktest/scoring_kaldi/best_wer 
# %WER 84.13 [ 14470 / 17200, 599 ins, 7390 del, 6481 sub ] run_imda_1a/exp/tri3/decode_wiz3ktest/wer_13_1.0
fi

# decode tri4
if [ ! -z $step18 ];then
    utils/mkgraph.sh $lang_test $exp_root/tri4 $exp_root/tri4/graph  || exit 1;
    for decode_set in  $test_set_2; do
     steps/decode_fmllr.sh --nj 10 --cmd "$train_cmd" --config conf/decode.config \
      $exp_root/tri4/graph $tgtdir/data/$decode_set $exp_root/tri4/decode_$decode_set  || exit 1;
    done
# # [md510@node06 lexicon_comparison]$ cat run_imda_1a/exp/tri4/decode_dev_imda_part3_ivr/scoring_kaldi/best_wer 
# %WER 40.00 [ 74222 / 185558, 12447 ins, 15054 del, 46721 sub ] run_imda_1a/exp/tri4/decode_dev_imda_part3_ivr/wer_15_0.5
fi

