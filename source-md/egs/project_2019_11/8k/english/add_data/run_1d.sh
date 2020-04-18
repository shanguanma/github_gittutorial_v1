#!/bin/bash

# 1d is same 1b, but I add boundarymic data to train_imda_part3_ivr_msf_i2r 
# the boundarymic data(e.g: I named it imda_boundarymic) is selected from /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/
# then I used the boundarymic date to do codec processing. then add to train_imda_part3_ivr_msf_i2r 

# the boundarymic data detail is as follows:
#1.  grep "boundarymic" /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/segments | awk '{sum += $4-$3}END{print sum/3600}'
#  425.296 


# dictionary is based on /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse
# then I using the biggist dictionaty(e.g:/home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs ) to solve oov. 
 
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


train_set=train_imda_part3_ivr_msf_i2r_imda_boundarymic_codec

test_set_1=wiz3ktest
test_set_2=dev_imda_part3_ivr
test_set_3=msf_baby_bonus-8k

tgtdir=run_1d
exp_root=$tgtdir/exp
add_data=imda_boundarymic
# 1. select boundarymic data from /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/

if [ ! -z $step01 ];then
  [ -d data/$add_data ] || mkdir -p data/$add_data
  grep "boundarymic" /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/segments >  data/$add_data/segments 
  head -n 10 data/$add_data/segments
  grep "boundarymic" /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/text >  data/$add_data/text
  head -n 10 data/$add_data/text
  grep "boundarymic" /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/wav.scp > data/$add_data/wav.scp
  head -n 10 data/$add_data/wav.scp
  grep "boundarymic" /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/train_imda_sept10/utt2spk > data/$add_data/utt2spk
  head -n 10  data/$add_data/utt2spk
  utils/utt2spk_to_spk2utt.pl  data/$add_data/utt2spk >  data/$add_data/spk2utt
  echo "cpmpute data length for data/$add_data/segments "
  awk '{sum += $4-$3}END{print sum/3600}' data/$add_data/segments 
  # it must have utt2dur file, beacuse trim and codec wav.scp case, kaldi can't get utt2dur.
  utils/data/get_utt2dur.sh data/$add_data
fi


# 2. make codec
# refrence: /home4/asr_resource/scripts/am/asr-training.sh  
#            /home3/zpz505/w2019/codec-augmented-from-ly/
# we use ffmpeg to downsample data to 8k HZ randomly using the codec provided in the codec list
src_add_data_dir=data/$add_data
tgt_add_data_dir=data/${add_data}_codec
tgt_add_data=${add_data}_codec
codec_list=source-md/egs/project_2019_11/8k/english/add_data/codec/codec-list-full.txt 
sampling_rate=8000


if [ ! -z $step02 ];then
  # 1. copy data to new folder 
  # this new folder is going to get codec data folder.
  utils/validate_data_dir.sh --no-feats  $src_add_data_dir
  [ -f $tgt_add_data_dir ] || rm -rf $tgt_add_data_dir
  utils/copy_data_dir.sh $src_add_data_dir $tgt_add_data_dir
  # 2. get trim wav.scp, new segments file
  source-md/egs/project_2019_11/8k/english/add_data/codec/trim-wav-scp.sh \
     $sampling_rate $tgt_add_data_dir 
   
fi
if [ ! -z $step03 ];then
   rm $tgt_add_data_dir/reco2dur
   utils/data/copy_data_dir.sh  $tgt_add_data_dir ${tgt_add_data_dir}/tmp
   # 3. modified wav.scp again
   sed -e 's/^/codec-/'  ${tgt_add_data_dir}/tmp/wav.scp > ${tgt_add_data_dir}/tmp/new_wav.scp || exit 1
   cat ${tgt_add_data_dir}/tmp/new_wav.scp | \
   source-md/egs/project_2019_11/8k/english/add_data/codec/add-codec-with-ffmpeg.pl \
     $sampling_rate $codec_list > ${tgt_add_data_dir}/tmp/wav.scp || exit 1;
   # 4. modified segments again
   cp ${tgt_add_data_dir}/segments ${tgt_add_data_dir}/tmp/new_segments
   cat ${tgt_add_data_dir}/tmp/new_segments |perl -ane 'use utf8; open qw(:std :utf8); chomp; m/(\S+)\s+(.*)/g or next; print "$1 codec-$2 $3 $4\n";' \
    > ${tgt_add_data_dir}/tmp/segments || exit 1
   utils/fix_data_dir.sh  ${tgt_add_data_dir}/tmp
   utils/data/copy_data_dir.sh  --utt-prefix "codec-" --spk-prefix "codec-" ${tgt_add_data_dir}/tmp ${tgt_add_data_dir} 
fi
   

# make mfcc for imda_boundarymic_codec
feat_dir=feat/mfcc
if [ ! -z $step04 ];then
   for part in $tgt_add_data; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
    utils/fix_data_dir.sh  data/$part
    utils/validate_data_dir.sh  data/$part 
  done
fi

# combine data
if [ ! -z $step05 ];then
    # get recod2dur file ,in order to get speed perturbed data
    #utils/data/get_reco2dur.sh data/$tgt_add_data
    utils/fix_data_dir.sh data/train_imda_part3_ivr_msf_i2r 
    utils/combine_data.sh  data/$train_set data/train_imda_part3_ivr_msf_i2r  data/$tgt_add_data
    utils/fix_data_dir.sh data/$train_set 
    utils/validate_data_dir.sh  data/$train_set
fi
# make new dictionary
if [ ! -z $step06 ];then
   [ -d $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data ] || mkdir -p $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data
   # 0. get oov
   cat data/train_imda_part3_ivr_msf_i2r_$tgt_add_data/text | source-md/egs/acumen/show-oov-with-count.pl --from=2  ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  > data/train_imda_part3_ivr_msf_i2r_$tgt_add_data/from_ntu_inhouse_oov
   #1. get oov word dict from biggest dictionary 
   source-md/egs/project_2019_11/8k/english/add_data/solve_oov_from_big_dict.py \
     data/train_imda_part3_ivr_msf_i2r_$tgt_add_data/from_ntu_inhouse_oov \
     /home4/hhx502/w2019/projects/ntu_16k_asr_aug22/data/local/dictv4.4_cs_ubs/lexicon.txt > test/oov_dict
   # 2. combine small dictionary
   cat ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  test/oov_dict | uniq > $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data/lexicon.txt
   # 3. show oov by using new dictionary
   cat data/train_imda_part3_ivr_msf_i2r_$tgt_add_data/text |  source-md/egs/acumen/show-oov-with-count.pl --from=2  $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data/lexicon.txt > data/train_imda_part3_ivr_msf_i2r_$tgt_add_data/from_ntu_inhouse_add_oov_list_$tgt_add_data_oov
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/extra_questions.txt  $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/optional_silence.txt $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/nonsilence_phones.txt $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data
   cp -r ../lexicon_comparison/run_ntu_inhouse_1a/data/dict_ntu_inhouse/silence_phones.txt  $tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data
fi

# make new lang
dictdir=$tgtdir/data/ntu_inhouse_add_oov_list_$tgt_add_data
lang=$tgtdir/data/lang
if [ ! -z $step07 ];then
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang


fi

# 3. I directly use tri4(e.g: it is from /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_ntu_inhouse_1a/exp/tri4) 
#  as ali file for these big data(e.g combine MSF_Baby_Bonus_Transcript ,sg-en-i2r  and train_imda_part3_ivr). 
#  so I do not train the gmm-hmm system from scratch 
# 
if [ ! -z $step08 ]; then
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
if [ ! -z $step09 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step10 ]; then
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

if [ ! -z $step11 ]; then
  utils/format_lm.sh $tgtdir/data/lang $tgtdir/data/local/lm/4gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi

# the purpose of this steps is in order to generate lang with the probability of silence
if [ ! -z $step12 ]; then
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


# because I don't want to do speed perturbed for imda_boundarymic_codec data, 
if [ ! -z $step13 ];then
   source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/nnet3/run_ivector_8k_1f.sh \
    --stage 3 \
    --train-set train_imda_part3_ivr_msf_i2r \
    --test-set-1 $test_set_1 \
    --test-set-2 $test_set_2 \
    --test-set-3 $test_set_3 \
    --test-set-is-run-mfcc-hires false  \
    --codec-data imda_boundarymic_codec \
    --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec \
    --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires \
    --nnet3-affix _1d

fi

if [ ! -z $step14 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1c.sh \
      --stage 6 \
      --train-stage 734 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $test_set_1 \
      --test-set-2  $test_set_2 \
      --test-set-3  $test_set_3 \
      --train-data-dir  data/train_imda_part3_ivr_msf_i2r_imda_boundarymic_codec \
      --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec \
      --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires  \
      --train-ivector-dir  data/nnet3_1d/ivectors_train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3_1d
# result 
# [md510@node06 add_data]$ cat run_1d/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/scoring_kaldi/best_wer 
# %WER 20.07 [ 37236 / 185558, 7304 ins, 8862 del, 21070 sub ] run_1d/exp/chain/tdnn_cnn_tdnnf_sp/decode_dev_imda_part3_ivr_better_pp/wer_9_0.5
# [md510@node06 add_data]$ cat run_1d/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/scoring_kaldi/best_wer 
# %WER 18.37 [ 6391 / 34794, 817 ins, 1481 del, 4093 sub ] run_1d/exp/chain/tdnn_cnn_tdnnf_sp/decode_msf_baby_bonus-8k_better_pp/wer_10_0.0
# [md510@node06 add_data]$ cat run_1d/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/scoring_kaldi/best_wer 
# %WER 38.51 [ 6624 / 17200, 912 ins, 2268 del, 3444 sub ] run_1d/exp/chain/tdnn_cnn_tdnnf_sp/decode_wiz3ktest_better_pp/wer_10_1.0

fi

# test ubs test data
# dev   set:   /home/hhx502/w2020/projects/ubs/data/deliverable/ubs2020_dev_man
#              /home/hhx502/w2020/projects/ubs/data/deliverable/ubs2020_dev_eng
#              /home/hhx502/w2020/projects/ubs/data/deliverable/ubs2020_dev_cs

# train set:  /home/hhx502/w2020/projects/ubs/data/deliverable/ubs2020_train
# dict : /home/hhx502/w2020/projects/ubs/data/deliverable/dictv0.1
# 1. make 40 dim mfcc, then get ivector
ubs2020_dev_man=ubs2020_dev_man
ubs2020_dev_eng=ubs2020_dev_eng
ubs2020_dev_cs=ubs2020_dev_cs
if [ ! -z $step15 ];then
   mfccdir=feat/mfcc_hires
  for dataset in $ubs2020_dev_man $ubs2020_dev_eng $ubs2020_dev_cs; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh /home/hhx502/w2020/projects/ubs/data/deliverable/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$cmd" --nj 10 --mfcc-config conf/mfcc_hires_8k.conf \
        data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
fi
# get test_set_3 ivector feature
if [ ! -z $step16 ]; then
  nnet3_affix=_1d
  for dataset in $ubs2020_dev_man $ubs2020_dev_eng $ubs2020_dev_cs; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      data/${dataset}_hires feat/nnet3${nnet3_affix}/extractor \
      data/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done
fi


if [ ! -z $step17 ];then
  source-md/egs/project_2019_11/8k/english/lexicon_comparison/local/chain/run_cnn_tdnnf_1c.sh \
      --stage 6 \
      --train-stage 734 \
      --get-egs-stage -10 \
      --gmm tri5 \
      --pp true \
      --first-lang $tgtdir/data/lang \
      --tgtdir $tgtdir \
      --train-set  $train_set \
      --test-set-1  $ubs2020_dev_man \
      --test-set-2  $ubs2020_dev_eng \
      --test-set-3  $ubs2020_dev_cs \
      --train-data-dir  data/train_imda_part3_ivr_msf_i2r_imda_boundarymic_codec \
      --lores-train-data-dir data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec \
      --train-data-sp-hires-dir data/train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires  \
      --train-ivector-dir  data/nnet3_1d/ivectors_train_imda_part3_ivr_msf_i2r_sp_imda_boundarymic_codec_hires \
      --test-set-dir   data \
      --test-set-ivector-dir data/nnet3_1d

# ubs2020_dev_man 97.92
# ubs2020_dev_eng 41.76
# ubs2020_dev_cs 91.62

fi 
