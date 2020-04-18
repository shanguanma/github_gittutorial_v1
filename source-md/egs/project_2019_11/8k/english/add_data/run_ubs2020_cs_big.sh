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


train_set=seame_ubs_train_cs200_cs_15_i2r_mandarin_hkust_dt_man200_imda_i2r_msf_imda_boundarymic_codec

test_set_1="wiz3ktest dev_imda_part3_ivr"
test_set_2="msf_baby_bonus-8k  ubs2020_dev_cs"
test_set_3="ubs2020_dev_eng ubs2020_dev_man"

tgtdir=run_ubs2020_cs_big
exp_root=$tgtdir/exp

# now using newest dictionary
srcdict=/home/zhb502/w2020/projects/ubs2020/data/local/dictv0.3
lang=$tgtdir/data/lang
dictdir=$tgtdir/data/ubs_dictv0.3
# make lang
if [ ! -z $step01 ];then
  [ -f $dictdir ] || mkdir -p $dictdir
  cp -r $srcdict/* $dictdir
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang

fi

feat_dir=feat/mfcc
if [ ! -z $step02 ];then
   utils/copy_data_dir.sh /home4/asr_resource/data/acoustic/mandarin/16k/i2r-mandarin  data/i2r-mandarin_8k
   utils/copy_data_dir.sh /home4/asr_resource/data/acoustic/mandarin/8k/hkust data/hkust_8k

   source-md/prepare_kaldi_data/make_downsample_for_pipe_hkust.py  data/hkust_8k/wav.scp> data/hkust_8k/wav_new.scp 
   head -n 10 data/hkust_8k/wav_new.scp
   mv data/hkust_8k/wav_new.scp data/hkust_8k/wav.scp
   source-md/prepare_kaldi_data/make_downsample_for_pipe.py data/i2r-mandarin_8k/wav.scp >  data/i2r-mandarin_8k/wav_new.scp
   head -n 10 data/i2r-mandarin_8k/wav_new.scp
   mv data/i2r-mandarin_8k/wav_new.scp data/i2r-mandarin_8k/wav.scp

   for part in hkust_8k i2r-mandarin_8k ; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
    utils/fix_data_dir.sh  data/$part
    utils/validate_data_dir.sh  data/$part
  done
 
fi
# dt_man200 is from /home/zhb502/w2020/projects/man_ubs_poc_jan13/data/dt_man200
#   corresponding to text is from /home/zhb502/w2020/projects/ubs2020/data/deliverable2_text/dt_man200-text 
#data/dt_man200/ is origin 8k data
if [ ! -z $step03 ];then
   utils/copy_data_dir.sh /home/zhb502/w2020/projects/man_ubs_poc_jan13/data/dt_man200 data/dt_man200_8k
   cat /home/zhb502/w2020/projects/ubs2020/data/deliverable2_text/dt_man200-text >data/dt_man200_8k/text 
   utils/fix_data_dir.sh  data/dt_man200_8k
   utils/validate_data_dir.sh --no-feats data/dt_man200_8k

   # cs_15_from_ASRUman500
   # /home/zhb502/w2020/projects/ubs2020/data/cs200_cs15 have downsample to 8k
   utils/copy_data_dir.sh /home/zhb502/w2020/projects/ubs2020/data/cs200_cs15 data/cs_15_8k_from_ASRUman500
   for  part in dt_man200_8k  cs_15_8k_from_ASRUman500; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
    utils/fix_data_dir.sh  data/$part
    utils/validate_data_dir.sh  data/$part
  done
fi



# combine data
if [ ! -z $step04 ];then
    # get recod2dur file ,in order to get speed perturbed data
    #utils/data/get_reco2dur.sh data/$tgt_add_data
    utils/fix_data_dir.sh data/ubs2020_train
    utils/combine_data.sh  data/$train_set \
           data/seame_trainset_8k_cs200_8k_ubs2020_train \
           data/cs_15_8k_from_ASRUman500 \
           data/dt_man200_8k \
           data/hkust_8k \
           data/i2r-mandarin_8k \
           data/train_imda_part3_ivr_msf_i2r_imda_boundarymic_codec 
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
if [ ! -z $step05 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step06 ]; then
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

if [ ! -z $step07 ]; then
  utils/format_lm.sh $tgtdir/data/lang $tgtdir/data/local/lm/4gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi


if [ ! -z $step08 ]; then
  steps/align_fmllr.sh --nj 40 --cmd "$cmd" \
    data/${train_set}  $lang \
    /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/run_ubs2020_cs/exp/tri4 \
    $exp_root/tri4_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 data/${train_set}  $lang $exp_root/tri4_ali $exp_root/tri5 || exit 1;

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
if [ ! -z $step10 ];then
  source-md/egs/project_2019_11/8k/english/add_data/run_ivector_common.sh \
      --stage 7 \
      --train-set seame_ubs_train_cs200_cs_15_i2r_mandarin_hkust_dt_man200_imda_i2r_msf_imda_boundarymic_codec \
      --nnet3-affix _ubs2020_cs_big 
fi
nnet3_affix=_ubs2020_cs_big
#ivector_train_set=${train_set}_sp
if [ ! -z $step11 ];then
  source-md/egs/project_2019_11/8k/english/add_data/run_cnn_tdnnf_1c.sh \
      --stage 5 \
      --train-stage 1556 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3${nnet3_affix}

fi


if [ ! -z $step12 ];then
  source-md/egs/project_2019_11/8k/english/add_data/run_cnn_tdnnf_1c.sh \
      --stage 16 \
      --train-stage 1556 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3${nnet3_affix}

fi

