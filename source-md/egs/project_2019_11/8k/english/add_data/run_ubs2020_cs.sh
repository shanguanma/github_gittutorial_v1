#!/bin/bash


. path.sh
. cmd.sh

cmd="slurm.pl  --quiet --exclude=node06,node05"
steps=
nj=80
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


train_set=seame_trainset_8k_cs200_8k_ubs2020_train

test_set_1="wiz3ktest dev_imda_part3_ivr"
test_set_2="msf_baby_bonus-8k  ubs2020_dev_cs"
test_set_3="ubs2020_dev_eng ubs2020_dev_man"

tgtdir=run_ubs2020_cs
exp_root=$tgtdir/exp

#srcdict=/home4/mtz503/w2020k/ubs-dict/langv0.2/dictv0.2
srcdict=/home/zhb502/w2020/projects/ubs2020/data/local/dictv0.21
lang=$tgtdir/data/lang
dictdir=$tgtdir/data/ubs_dictv0.21
#ubs_train_set=/home/hhx502/w2020/projects/ubs/data/deliverable/ubs2020_train
# make lang
if [ ! -z $step01 ];then
  [ -f $dictdir ] || mkdir -p $dictdir
  cp -r $srcdict/* $dictdir
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang

fi



# make mfcc 
# data/seame_trainset is from /home4/md510/w2018/data/seame/train 
feat_dir=feat/mfcc
if [ ! -z $step02 ];then 
   utils/copy_data_dir.sh data/cs200 data/cs200_8k
   utils/copy_data_dir.sh data/seame_trainset data/seame_trainset_8k
   source-md/prepare_kaldi_data/make_downsample_for_pipe.py  data/cs200_8k/wav.scp > data/cs200_8k/wav_new.scp
   head -n 5 data/cs200_8k/wav_new.scp
   mv data/cs200_8k/wav_new.scp data/cs200_8k/wav.scp 
   source-md/prepare_kaldi_data/make_downsample_for_pipe.py data/seame_trainset_8k/wav.scp > data/seame_trainset_8k/wav_new.scp
   head -n 5  data/seame_trainset_8k/wav_new.scp
   mv  data/seame_trainset_8k/wav_new.scp  data/seame_trainset_8k/wav.scp
   for part in cs200_8k seame_trainset_8k ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
    utils/fix_data_dir.sh  data/$part
    utils/validate_data_dir.sh  data/$part
  done
fi

# combine data
if [ ! -z $step03 ];then
    # get recod2dur file ,in order to get speed perturbed data
    #utils/data/get_reco2dur.sh data/$tgt_add_data
    utils/fix_data_dir.sh data/ubs2020_train
    utils/combine_data.sh  data/$train_set data/seame_trainset_8k data/cs200_8k  data/ubs2020_train
    utils/fix_data_dir.sh data/$train_set
    utils/validate_data_dir.sh  data/$train_set
fi

###############################################################################
# pepared new lang_test 
# more detail ,you can read /home4/md510/w2019a/kaldi-recipe/lm/data/lm/perplexities.txt
#                           /home4/md510/w2019a/kaldi-recipe/lm/data/lm/ 
#                           /home4/md510/w2019a/kaldi-recipe/lm/train_lms_srilm.sh 
###############################################################################
datadir=data
lmdir=$tgtdir/data/local/lm
train_data=${datadir}/$train_set
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



train_data=data/$train_set
datadir=data
if [ ! -z $step07 ]; then
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
if [ ! -z $step08 ]; then
  steps/train_mono.sh --nj  $nj  --cmd "$cmd" \
    $datadir/${train_set}_30kshort $lang $exp_root/mono || exit 1
fi

if [ ! -z $step09 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set}_200k_nodup $lang $exp_root/mono $exp_root/mono_ali || exit 1

  steps/train_deltas.sh --cmd "$cmd" \
    2500 20000 $datadir/${train_set}_200k_nodup $lang $exp_root/mono_ali $exp_root/tri1 || exit 1

fi

if [ ! -z $step10 ]; then
  # The 100k_nodup data is used in the nnet2 recipe.
  steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                    $datadir/${train_set}_200k_nodup  $lang $exp_root/tri1 $exp_root/tri1_ali_200k_nodup

  # From now, we start using all of the data (except some duplicates of common
  # utterances, which don't really contribute much).
  steps/align_si.sh --nj $nj --cmd "$cmd" \
   $datadir/${train_set}_nodup $lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
    2500 20000 $datadir/${train_set}_nodup $lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

fi
if [ ! -z $step11 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set}_nodup $lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 $datadir/${train_set}_nodup $lang $exp_root/tri2_ali $exp_root/tri3 || exit 1;

fi

if [ ! -z $step12 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$cmd" \
    $datadir/${train_set}_nodup  $lang $exp_root/tri3 $exp_root/tri3_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 $datadir/${train_set}_nodup  $lang $exp_root/tri3_ali $exp_root/tri4 || exit 1;

fi

# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step13 ]; then
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

# make i-vector

# 1. get lower train data, in order to get ali, it is used as input data of steps/align_fmllr_lats.sh  
# it is 13 dim mfcc 
lores_train_data_dir=data/seame_trainset_8k_cs200_8k_sp_ubs2020_train
if [ ! -z $step15 ];then
   for datadir in seame_trainset_8k cs200_8k; do
      utils/data/perturb_data_dir_speed_3way.sh data/${datadir} data/${datadir}_sp
      utils/fix_data_dir.sh data/${datadir}_sp

      mfccdir=feat/mfcc_perturbed
      steps/make_mfcc.sh --cmd "$cmd" --nj 40 --mfcc-config conf/mfcc_8k.conf \
        data/${datadir}_sp feat_log/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh \
        data/${datadir}_sp feat_log/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_sp
    done 
   utils/combine_data.sh  data/seame_trainset_8k_cs200_8k_sp  data/seame_trainset_8k_sp data/cs200_8k_sp
   utils/validate_data_dir.sh data/seame_trainset_8k_cs200_8k_sp

   utils/combine_data.sh $lores_train_data_dir data/seame_trainset_8k_cs200_8k_sp data/ubs2020_train
   utils/validate_data_dir.sh  $lores_train_data_dir 
fi




#  get 40 dim mfcc  for seame_trainset_8k and cs200_8k
# becacuse ubs_trainset has got 40 dim  mfcc, so i don't do it again.
if [ ! -z $step16 ];then
  mfccdir=feat/mfcc_hires
  for dataset in cs200_8k_sp seame_trainset_8k_sp; do
    utils/copy_data_dir.sh  data/$dataset data/${dataset}_hires
    utils/data/perturb_data_dir_volume.sh data/${dataset}_hires 
    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$train_cmd" data/${dataset}_hires feat_log/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires feat_log/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done
fi

# $train_data_sp_hires_dir is 40 dim mfcc , it is used to train ivector extrator
#                                           it is used as input data of chain model.
train_data_sp_hires_dir=data/seame_trainset_8k_cs200_8k_sp_hires_ubs2020_train_hires
if [ ! -z $step17 ];then
   utils/combine_data.sh data/seame_trainset_8k_cs200_8k_sp_hires data/seame_trainset_8k_sp_hires data/cs200_8k_sp_hires
   utils/validate_data_dir.sh  data/seame_trainset_8k_cs200_8k_sp_hires

   utils/combine_data.sh  $train_data_sp_hires_dir  data/seame_trainset_8k_cs200_8k_sp_hires data/ubs2020_train_hires
   utils/validate_data_dir.sh $train_data_sp_hires_dir

fi
nnet3_affix=_ubs2020_cs
# ivector extractor training
if [ ! -z $step18 ]; then
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    $train_data_sp_hires_dir \
    feat/nnet3${nnet3_affix}/pca_transform
fi

if [ ! -z $step19 ]; then
  steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj 50 --num-frames 700000 \
    $train_data_sp_hires_dir 512 \
    feat/nnet3${nnet3_affix}/pca_transform feat/nnet3${nnet3_affix}/diag_ubm
fi

if [ ! -z $step20 ]; then
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj 10 \
    $train_data_sp_hires_dir feat/nnet3${nnet3_affix}/diag_ubm \
    feat/nnet3${nnet3_affix}/extractor || exit 1;
fi
train_data_sp_hires=seame_trainset_8k_cs200_8k_sp_hires_ubs2020_train_hires
if [ ! -z $step21 ]; then
  # We extract iVectors on all the ${train_set} data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $train_data_sp_hires_dir ${train_data_sp_hires_dir}_max2_hires

  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 30 \
    ${train_data_sp_hires_dir}_max2_hires feat/nnet3${nnet3_affix}/extractor \
    data/nnet3${nnet3_affix}/ivectors_${train_data_sp_hires}  || exit 1;
fi

# because $test_set_1 $test_set_2 $test_set_3 have get 40 dim mfcc, so I don't do it again here.
if [ ! -z $step22 ]; then
  for dataset in $test_set_1 $test_set_2 $test_set_3; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      data/${dataset}_hires feat/nnet3${nnet3_affix}/extractor \
      data/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done
fi

train_data_dir=data/$train_set  # it is used to build chain tree 
train_ivector_dir=data/nnet3${nnet3_affix}/ivectors_${train_data_sp_hires} # it is used as train set ivector feature of chain model
test_set_dir=data
test_set_ivector_dir=data/nnet3${nnet3_affix} 
if [ ! -z $step23 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1f.sh \
      --stage 8 \
      --train-stage 378 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  $train_data_dir \
      --lores-train-data-dir  $lores_train_data_dir \
      --train-data-sp-hires-dir $train_data_sp_hires_dir \
      --train-ivector-dir $train_ivector_dir \
      --test-set-dir $test_set_dir \
      --test-set-ivector-dir $test_set_ivector_dir || exit 1;
fi

# result 
# [md510@node07 add_data]$ cat  run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/scoring_kaldi/best_wer 
# %WER 46.03 [ 85412 / 185558, 12883 ins, 16632 del, 55897 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/wer_9_0.5
# [md510@node07 add_data]$ cat  run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/scoring_kaldi/best_wer 
# %WER 39.75 [ 13829 / 34794, 1091 ins, 3260 del, 9478 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/wer_9_0.0
# [md510@node07 add_data]$ cat  run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_better_pp/scoring_kaldi/best_wer 
# %WER 31.79 [ 13230 / 41622, 1575 ins, 3695 del, 7960 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_better_pp/wer_8_1.0
# [md510@node07 add_data]$ cat  run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_eng_better_pp/scoring_kaldi/best_wer 
# %WER 29.74 [ 8770 / 29486, 975 ins, 2708 del, 5087 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_eng_better_pp/wer_9_0.5
# [md510@node07 add_data]$ cat  run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_better_pp/scoring_kaldi/best_wer 
# %WER 32.30 [ 8559 / 26501, 907 ins, 2873 del, 4779 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_better_pp/wer_9_1.0
# [md510@node07 add_data]$ cat  run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
# %WER 47.18 [ 8115 / 17200, 1206 ins, 1928 del, 4981 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/wer_11_0.0



if [ ! -z $step24 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1f.sh \
      --stage 9 \
      --train-stage 378 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  $train_data_dir \
      --lores-train-data-dir  $lores_train_data_dir \
      --train-data-sp-hires-dir $train_data_sp_hires_dir \
      --train-ivector-dir $train_ivector_dir \
      --test-set-dir $test_set_dir \
      --test-set-ivector-dir $test_set_ivector_dir || exit 1;
fi
# it can't make graph
# error log
# cat log/run_ubs2020_cs/steps24.log 


if [ ! -z $step25 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1f.sh \
      --stage 11 \
      --train-stage 378 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  $train_data_dir \
      --lores-train-data-dir  $lores_train_data_dir \
      --train-data-sp-hires-dir $train_data_sp_hires_dir \
      --train-ivector-dir $train_ivector_dir \
      --test-set-dir $test_set_dir \
      --test-set-ivector-dir $test_set_ivector_dir || exit 1;

# result 
# [md510@node07 add_data]$  cat run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_mandarin_pp/scoring_kaldi/best_wer 
#%WER 33.92 [ 14119 / 41622, 1672 ins, 3633 del, 8814 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_mandarin_pp/wer_9_0.0
#[md510@node07 add_data]$  cat run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_mao_mandarin_pp/scoring_kaldi/best_wer 
#%WER 32.67 [ 8657 / 26501, 963 ins, 2567 del, 5127 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_man_mao_mandarin_pp/wer_10_0.0
fi

# I use mao's cs 4-gram lm  to make HCLG.fst
# First, /home4/mtz503/w2020k/4-gram-ubs2020/md-4gram-test/lm4.gz
# second, /home4/mtz503/w2020k/4-gram-ubs2020/md-4gram-test/lm4-u-s-u-cs2-cs1.gz
# third /home4/mtz503/w2020k/4-gram-ubs2020/md-4gram-test/lm4-u-h1-m1.gz

if [ ! -z $step26 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1f.sh \
      --stage 13 \
      --train-stage 378 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  $train_data_dir \
      --lores-train-data-dir  $lores_train_data_dir \
      --train-data-sp-hires-dir $train_data_sp_hires_dir \
      --train-ivector-dir $train_ivector_dir \
      --test-set-dir $test_set_dir \
      --test-set-ivector-dir $test_set_ivector_dir || exit 1;

# results:
# [md510@node07 add_data]$ cat run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_cs1_pp/scoring_kaldi/best_wer 
#%WER 31.40 [ 13068 / 41622, 1605 ins, 3630 del, 7833 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_cs1_pp/wer_10_0.0
#[md510@node07 add_data]$ cat run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_cs2_pp/scoring_kaldi/best_wer 
#%WER 31.26 [ 13012 / 41622, 1480 ins, 3690 del, 7842 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_cs2_pp/wer_9_0.5
#[md510@node07 add_data]$ cat run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_cs3_pp/scoring_kaldi/best_wer 
#%WER 33.30 [ 13860 / 41622, 1312 ins, 4312 del, 8236 sub ] run_ubs2020_cs/exp/chain/tdnn_cnn_tdnnf_sp/decode_ubs2020_dev_cs_mao_cs3_pp/wer_10_0.0

fi
# steps 19-22
if [ ! -z $step27 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1f.sh \
      --stage 22 \
      --train-stage 378 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  $train_data_dir \
      --lores-train-data-dir  $lores_train_data_dir \
      --train-data-sp-hires-dir $train_data_sp_hires_dir \
      --train-ivector-dir $train_ivector_dir \
      --test-set-dir $test_set_dir \
      --test-set-ivector-dir $test_set_ivector_dir || exit 1;

fi

# 2020/2/21, I use mao's lm with adaption model. then make HCLG.fst
# pure english 4gram: /home4/mtz503/w2020k/4-gram-ubs2020/english-4gram/lm4-kn-adapt0.8.gz
# pure mandarin 4gram:/home4/mtz503/w2020k/4-gram-ubs2020/mandarin-4gram/lm4-kn-adapt0.8-161.gz
# the biggest 4gram:(text:cn/en/cs) /home4/mtz503/w2020k/4-gram-ubs2020/lm4-bigbig/text-1/lm4-kn-adapt-0.8_pruned.gz 
# steps 23-28
if [ ! -z $step28 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1f.sh \
      --stage 28 \
      --train-stage 378 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  $train_data_dir \
      --lores-train-data-dir  $lores_train_data_dir \
      --train-data-sp-hires-dir $train_data_sp_hires_dir \
      --train-ivector-dir $train_ivector_dir \
      --test-set-dir $test_set_dir \
      --test-set-ivector-dir $test_set_ivector_dir || exit 1;

fi

