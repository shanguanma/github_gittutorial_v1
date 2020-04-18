#!/bin/bash


. path.sh
. cmd.sh

cmd="slurm.pl  --quiet --exclude=node07,node08"
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


train_set=ubs2020_train

test_set_1="wiz3ktest dev_imda_part3_ivr"
test_set_2="msf_baby_bonus-8k  ubs2020_dev_cs"
test_set_3="ubs2020_dev_eng ubs2020_dev_man"

tgtdir=run_ubs2020_only_ubs_cs
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


###############################################################################
# GMM system training using add augment seame supervised data
###############################################################################
exp_root=$tgtdir/exp
if [ ! -z $step08 ]; then
  steps/train_mono.sh --nj  $nj  --cmd "$cmd" \
    $datadir/${train_set} $lang $exp_root/mono || exit 1
fi

if [ ! -z $step09 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set} $lang $exp_root/mono $exp_root/mono_ali || exit 1

  steps/train_deltas.sh --cmd "$cmd" \
    2500 20000 $datadir/${train_set} $lang $exp_root/mono_ali $exp_root/tri1 || exit 1

fi

if [ ! -z $step10 ]; then
  # From now, we start using all of the data (except some duplicates of common
  # utterances, which don't really contribute much).
  steps/align_si.sh --nj $nj --cmd "$cmd" \
   $datadir/${train_set} $lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
    2500 20000 $datadir/${train_set} $lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

fi
if [ ! -z $step11 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $datadir/${train_set} $lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 $datadir/${train_set} $lang $exp_root/tri2_ali $exp_root/tri3 || exit 1;

fi

if [ ! -z $step12 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$cmd" \
    $datadir/${train_set}  $lang $exp_root/tri3 $exp_root/tri3_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 $datadir/${train_set}  $lang $exp_root/tri3_ali $exp_root/tri4 || exit 1;

fi
# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step13 ]; then
   # Now we compute the pronunciation and silence probabilities from training data,
   # and re-create the lang directory.
   steps/get_prons.sh --cmd "$cmd" \
    $datadir/${train_set} $lang $exp_root/tri4

   utils/dict_dir_add_pronprobs.sh --max-normalize true \
    $dictdir \
    $exp_root/tri4/pron_counts_nowb.txt $exp_root/tri4/sil_counts_nowb.txt \
    $exp_root/tri4/pron_bigram_counts_nowb.txt $tgtdir/data/local/dict_pp

   utils/prepare_lang.sh $tgtdir/data/local/dict_pp \
    "<unk>" $tgtdir/data/lang_pp/tmp $tgtdir/data/lang_pp
   utils/format_lm.sh $tgtdir/data/lang_pp $tgtdir/data/local/lm/4gram.me.gz \
     $dictdir/lexiconp.txt $tgtdir/data/lang_test_pp

fi
# 1. get lower train data, in order to get ali, it is used as input data of steps/align_fmllr_lats.sh  
# it is 13 dim mfcc 
lores_train_data_dir=data/${train_set}_sp
if [ ! -z $step15 ];then
   for datadir in ubs2020_train ; do
      utils/data/perturb_data_dir_speed_3way.sh data/${datadir} data/${datadir}_sp
      utils/fix_data_dir.sh data/${datadir}_sp

      mfccdir=feat/mfcc_perturbed
      steps/make_mfcc.sh --cmd "$cmd" --nj 40 --mfcc-config conf/mfcc_8k.conf \
        data/${datadir}_sp feat_log/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      steps/compute_cmvn_stats.sh \
        data/${datadir}_sp feat_log/make_mfcc/${datadir}_sp $mfccdir || exit 1;
      utils/fix_data_dir.sh data/${datadir}_sp
    done
fi




#  get 40 dim mfcc
train_data_sp_hires_dir=data/${train_set}_sp_hires 
if [ ! -z $step16 ];then
  mfccdir=feat/mfcc_hires
  for dataset in ${train_set}_sp ; do
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

nnet3_affix=_ubs2020_only_ubs_cs
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
train_data_sp_hires=${train_set}_sp_hires
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
if [ ! -z $step23 ];then
  source-md/egs/project_2019_11/8k/english/add_data/run_cnn_tdnnf_1d.sh \
      --stage 6 \
      --train-stage -10 \
      --get-egs-stage -10 \
      --gmm tri4 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3${nnet3_affix}

fi

