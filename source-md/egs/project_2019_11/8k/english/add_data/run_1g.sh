#!/bin/bash

# prepare data


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

feat_dir=feat/mfcc
if [ ! -z $step01 ];then
   # make 13 dimension mfcc for fishereng_100h( it is about 100hours),
   # it is used to train gmm-hmm system.
   # it is select from fishereng(it is about 1900 hours).
   # how to selcet ?
   # utils/subset_data_dir.sh --per-spk /home4/asr_resource/data/acoustic/english/8k/fishereng  9 data/fishereng_100h
   for part in fishereng_100h SWB1_50h; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
    utils/fix_data_dir.sh  data/$part
    utils/validate_data_dir.sh  data/$part
  done
fi

if [ ! -z $step02 ]; then
    # Although the nnet will be trained by high resolution data
    # _sp stands for speed-perturbed
    # it is 13 dimension mfcc 
    # it is used as input data to steps/align_fmllr_lats.sh 
    # # Version of align_fmllr.sh that generates lattices (lat.*.gz) with
    # alignments of alternative pronunciations in them.  Mainly intended
    # as a precursor to LF-MMI/chain training for now. 
    for datadir in fishereng_100h SWB1_50h; do
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

if [ ! -z $step03 ]; then
  # it is 40 dimension mfcc
  # it is used as input to chain model.
  # it is used as input to train ivector extractor.
  mfccdir=feat/mfcc_hires
  
  for dataset in fishereng_100h SWB1_50h; do
    utils/copy_data_dir.sh data/${dataset}_sp data/${dataset}_sp_hires
    utils/data/perturb_data_dir_volume.sh data/${dataset}_sp_hires

    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$cmd" data/${dataset}_sp_hires feat_log/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_sp_hires feat_log/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_sp_hires;
  done

fi

# add codec for seame set and cs200(it is from datatang ASRU2019 )
# cs200 : /home4/md510/w2018/data/asru2019/kaldi_format_data/cs200 
# seame : /home4/md510/w2018/data/seame

if [ ! -z $step04 ];then
  utils/copy_data_dir.sh /home4/md510/w2018/data/asru2019/kaldi_format_data/cs200  data/cs200
  utils/copy_data_dir.sh /home4/md510/w2018/data/seame/train data/seame_trainset
  for dataset in cs200 seame_trainset; do
   source-md/egs/project_2019_11/8k/english/add_data/codec/add-codec.sh \
     --steps 1-2 \
     data/$dataset data/${dataset}_codec
  done
  
fi 


feat_dir=feat/mfcc
if [ ! -z $step05 ];then
   # it is used to train gmm-hmm system.
   # it is 13 dim mfcc
   for part in cs200_codec seame_trainset_codec; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 40  --mfcc-config conf/mfcc_8k.conf data/$part feat_log/make_mfcc/$part $feat_dir
    steps/compute_cmvn_stats.sh data/$part feat_log/make_mfcc/$part $feat_dir
    utils/fix_data_dir.sh  data/$part
    utils/validate_data_dir.sh  data/$part
  done
fi


if [ ! -z $step06 ]; then
  # it is 40 dimension mfcc
  # it is used as input to chain model.
  # it is used as input to train ivector extractor.
  mfccdir=feat/mfcc_hires

  for dataset in cs200_codec seame_trainset_codec; do
    utils/copy_data_dir.sh data/${dataset} data/${dataset}_hires
    steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
        --cmd "$cmd" data/${dataset}_hires feat_log/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires feat_log/make_hires/${dataset} $mfccdir;

    # Remove the small number of utterances that couldn't be extracted for some
    # reason (e.g. too short; no such file).
    utils/fix_data_dir.sh data/${dataset}_hires;
  done

fi

