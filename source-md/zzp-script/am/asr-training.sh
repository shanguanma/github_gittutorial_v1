#!/bin/bash

# Copyright 2018-2020 (Authors: zeng zhiping zengzp0912@gmail.com) 2020-03-18 updated
# This script built a nnet3 chain system,
# you can add noise, codecs in feat extraction steps
#

. ./path.sh

echo
echo "## LOG: $0 $@"
echo

# begin option
export SCRIPT_ROOT='/home4/mtz503/w2020k/w-script/w2020k/zzp-script/am'
cmd="slurm.pl --quiet --nodelist=node05"
nj=100
steps=

#GMM
state_num=10000
pdf_num=200000
train_gmm=true ##  if false, please add "--gsdir $expdir/tri4a"
gmm_steps=1-11 ##  train GMM default --steps 1-11  1.mono 2.ali 3. tri1 4.ali  5.tri2 6.ali  7.tri3 8.ali 9.lang-silprob 10.tri4 11.ali

# TDNN Chain
numleaves=14000
hidden_dim=1024
bottleneck_dim=256
small_dim=512
chainname=chain${hidden_dim}tdnnf
num_jobs_initial=4
num_jobs_final=4
tdnn_steps=1-7
train_stage=-10

# feat related
sp=''  # if not empty, do speed_3way perturb
sampling_rate=8000 #16000
trim_wav_scp=true
mfcc_hires_conf=conf/mfcc_hires_8k.conf #conf/mfcc_hires_8k.conf
mfcc_conf=conf/mfcc_8k.conf #conf/mfcc-8k.conf
# add noise
add_noise=true  # if not empty, add noise
noise_list="reverb music noise babble"
subset_noise_rate=0.25 # 0.25 * 4(noise_list) = 1 replications
rir_dir=/home4/asr_resource/data/noise/RIRS_NOISES # don't need sample rate same with source data
musan_noise_dir=/home4/asr_resource/data/noise/musan # need sample rate same with source data
# add codecs
do_codecs=true
subset_codec_rate=0.5  # 0.5 half of all data(combine clean and noise data first)
codec_list_file=${SCRIPT_ROOT}/common/codec/codec-list.txt

# ivector_extractor
train_ivector_extractor=true

# Experiment location
tgtdir=/home/mtz503/w2020k/projects/zp-work
train_data=
dev_man=
dev_sge=
dictdir=
lang=
expdir=
gsdir=
alidir=
nnetdir=
ivector_extractor=
graph=

# ngram language model
ngram=3

# end option

. parse_options.sh || (echo "make sure the location you run this script has directory: steps utils" && exit 1)

function Example {
 cat<<EOF

 $(basename $0) [options] <source-dict> <src-train-data> <src-test-data1> <src-test-data2>

 [options]:
 --SCRIPT_ROOT                    # value, "$SCRIPT_ROOT"
 --cmd                            # value, "$cmd"
 --nj                             # value, $nj
 --steps                          # value, "$steps", for instance, "--steps 1,2,3,4"

 # GMM
 --state_num                      # value, "$state_num"
 --pdf_num                        # value, "$pdf_num"
 --train_gmm                      # value, "$train_gmm"
 --gmm_steps                      # value, "$gmm_steps"

 # TDNN Chain
 --numleaves                      # value, "$numleaves"
 --hidden_dim                     # value, "$hidden_dim"
 --bottleneck_dim                 # value, "$bottleneck_dim"
 --small_dim                      # value, "$small_dim"
 --chainname                      # value, "$chainname"
 --num_jobs_initial               # value, "$num_jobs_initial"
 --num_jobs_final                 # value, "$num_jobs_final"
 --tdnn_steps                     # value, $tdnn_steps
 --train_stage                    # value, $train_stage

 # Feat
 --sp                             # value, "$sp"
 --sampling_rate                  # value, $sampling_rate
 --trim_wav_scp                   # value, $trim_wav_scp
 --mfcc_hires_conf                # value, $mfcc_hires_conf
 --mfcc_conf                      # value, $mfcc_conf
 # noise
 --add_noise                      # value, $add_noise
 --noise_list                     # value, "$noise_list"
 --subset_noise_rate              # value, $subset_noise_rate
 --rir_dir                        # value, $rir_dir
 --musan_noise_dir                # value, $musan_noise_dir
 # codecs
 --do_codecs                      # value, $do_codecs
 --subset_codec_rate              # value, $subset_codec_rate
 --codec_list_file                # value, "$codec_list_file"

 # ivector_extractor
 --train_ivector_extractor        # value, "$train_ivector_extractor"

 # Exp location
 --tgtdir                         # value, $tgtdir
 --train_data                     # value, $train_data
 --dev_man                        # value, $dev_man
 --dev_sge                        # value, $dev_sge
 --dictdir                        # value, $dictdir
 --lang                           # value, $lang
 --expdir                         # value, $expdir
 --gsdir                          # value, $gsdir
 --alidir                         # value, $alidir
 --nnetdir                        # value, $nnetdir
 --ivector_extractor              # value, $ivector_extractor
 --graph                          # value, $graph


 [steps]:
 1: copy data
 2: add noise
 3: add codecs
 4: mfcc feat extraction
 5: copy dict
 6: prepare lang
 7: training GMM (default, if --train_gmm=true)
 8: alignment & generate lattice (if --train_gmm=false)
 9-11: training ivector extractor (default, --train_ivector_extractor=true, if false, need metion "--ivector_extractor" location )
 12: extraction train_ivector
 13: TDNN training
 14: ngram lm, G.fst
 15: mkgraph
 16: decoding dev_man
 17: decoding dev_sge

 [examples]:
 # baseline
 sbatch -o /home/zpz505/w2019/seame/log/step01-19.log \
 $0 --steps 1-19 --trim_wav_scp false --tgtdir /home/zpz505/w2019/seame/baseline \
  /home/zpz505/w2019/seame/source-dict \
  /home/zpz505/w2019/seame/source-data/train_man_c_eng_w \
  /home/zpz505/w2019/seame/source-data/dev-man \
  /home/zpz505/w2019/seame/source-data/dev-sge

 # baseline + sp
 sbatch -o /home/zpz505/w2019/seame/log/sp-step01-19.log \
 $0 --steps 1-19 --sp "_sp" --trim_wav_scp false --tgtdir /home/zpz505/w2019/seame/baseline-sp \
  /home/zpz505/w2019/seame/source-dict \
  /home/zpz505/w2019/seame/source-data/train_man_c_eng_w \
  /home/zpz505/w2019/seame/source-data/dev-man \
  /home/zpz505/w2019/seame/source-data/dev-sge

 # baseline + noise
 sbatch -o /home/zpz505/w2019/seame/log/noise-step01-19.log \
 $0 --steps 1-19 --add_noise true --trim_wav_scp false --tgtdir /home/zpz505/w2019/seame/baseline-noise \
  /home/zpz505/w2019/seame/source-dict \
  /home/zpz505/w2019/seame/source-data/train_man_c_eng_w \
  /home/zpz505/w2019/seame/source-data/dev-man \
  /home/zpz505/w2019/seame/source-data/dev-sge

 # baseline + sp + noise
 sbatch -o /home/zpz505/w2019/seame/log/noise-sp-trim-step01-19.log \
 $0 --steps 1-19 --sp "_sp" --add_noise true --trim_wav_scp true --tgtdir /home/zpz505/w2019/seame/baseline-noise-sp-trim \
  /home/zpz505/w2019/seame/source-dict \
  /home/zpz505/w2019/seame/source-data/train_man_c_eng_w \
  /home/zpz505/w2019/seame/source-data/dev-man \
  /home/zpz505/w2019/seame/source-data/dev-sge

 # baseline + sp + codecs
 sbatch -o /home/zpz505/w2019/seame/log/sp-trim-codecs-step01-19.log \
 $0 --steps 1-19 --sp "_sp" --sampling_rate 8000 --do_codecs true --trim_wav_scp true \
  --mfcc_hires_conf conf/mfcc_hires_8k.conf --mfcc_conf conf/mfcc-8k.conf \
  --tgtdir /home/zpz505/w2019/seame/baseline-sp-trim-codecs \
  /home/zpz505/w2019/seame/source-dict \
  /home/zpz505/w2019/seame/source-data/train_man_c_eng_w \
  /home/zpz505/w2019/seame/source-data/dev-man \
  /home/zpz505/w2019/seame/source-data/dev-sge

 # baseline + sp + noise + codecs
 sbatch -o /home/zpz505/w2019/seame/log/noise-sp-trim-codecs-step01-19.log \
 $0 --steps 1-19 --sp "_sp" --sampling_rate 8000 --do_codecs true --add_noise true --trim_wav_scp true \
  --mfcc_hires_conf conf/mfcc_hires_8k.conf --mfcc_conf conf/mfcc-8k.conf \
  --musan_noise_dir /home4/asr_resource/data/noise/musan_down_to_8k \
  --tgtdir /home/zpz505/w2019/seame/baseline-noise-sp-trim-codecs \
  /home/zpz505/w2019/seame/source-dict \
  /home/zpz505/w2019/seame/source-data/train_man_c_eng_w \
  /home/zpz505/w2019/seame/source-data/dev-man \
  /home/zpz505/w2019/seame/source-data/dev-sge

EOF
}

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

if [ ! -z "$tgtdir" ]; then
  if [ -z "$train_data" ]; then   declare train_data=${tgtdir}/data/train ;  fi
  if [ -z "$dev_man" ]; then   declare dev_man=$tgtdir/data/dev-man ;  fi
  if [ -z "$dev_sge" ]; then   declare dev_sge=$tgtdir/data/dev-sge ;  fi
  if [ -z "$dictdir" ]; then   declare dictdir=$tgtdir/data/local/dict ;  fi
  if [ -z "$lang" ]; then   declare lang=$tgtdir/data/lang ;  fi
  if [ -z "$expdir" ]; then   declare expdir=$tgtdir/exp ;  fi
  if [ ! -z "$expdir" ]; then
    if [ -z "$gsdir" ]; then   declare gsdir=$expdir/tri4a ;  fi
    if [ -z "$alidir" ]; then   declare alidir=$expdir/tri4a/ali_train ;  fi
    if [ -z "$nnetdir" ]; then   declare nnetdir=$expdir/tdnn ;  fi
    if [ ! -z "$nnetdir" ]; then
      if [ -z "$ivector_extractor" ]; then   declare ivector_extractor=$nnetdir/ivector-extractor ;  fi
      if [ -z "$graph" ]; then   declare graph=$nnetdir/${chainname}/graph ;  fi
    fi
  fi
fi

if [ $# -ne 4 ]; then
  Example && exit 1
fi


src_dict=$1
src_train=$2
src_dev_man=$3
src_dev_sge=$4


#########################################################################################
##                         copy data from source to target directory                   ##
#########################################################################################
if [ ! -z $step01 ]; then
  [ -d $tgtdir/data ] || mkdir -p $tgtdir/data
  utils/data/copy_data_dir.sh  $src_train $train_data || exit 1;
  utils/data/copy_data_dir.sh  $src_dev_man $dev_man || exit 1;
  utils/data/resample_data_dir.sh $sampling_rate $dev_man || exit 1;
  utils/data/copy_data_dir.sh  $src_dev_sge $dev_sge || exit 1;
  utils/data/resample_data_dir.sh $sampling_rate $dev_sge || exit 1;
  if [ ! -z ${sp} ]; then
    utils/data/perturb_data_dir_speed_3way.sh $train_data ${train_data}_sp || exit 1;
    if [ "${trim_wav_scp}" == "true" ]; then
      utils/data/get_utt2dur.sh ${train_data}_sp || exit 1;
      ${SCRIPT_ROOT}/common/codec/trim-wav-scp.sh $sampling_rate ${train_data}_sp || exit 1;
      echo 'rewrite segments and wav.scp add trim done!'
    else
      if [ "${do_codecs}" == "true" ] || [ "${add_noise}" == "true" ]; then
        echo "WARNNING: need --trim_wav_scp true " && exit 1;
      fi
    fi
  else
    if [ "${trim_wav_scp}" == "true" ]; then
      utils/data/get_utt2dur.sh ${train_data} || exit 1;
      ${SCRIPT_ROOT}/common/codec/trim-wav-scp.sh $sampling_rate ${train_data} || exit 1;
      echo 'rewrite segments and wav.scp add trim done!'
    else
      if [ "${do_codecs}" == "true" ] || [ "${add_noise}" == "true" ]; then
        echo "WARNNING: need --trim_wav_scp true " && exit 1;
      fi
    fi
  fi
  echo "## LOG (step01): copy and prepare data from source to '$train_data' done!"
fi

if [ ! -z ${sp} ]; then
    train_data=${train_data}_sp
    echo "Using train data 3way speeh perturb"
fi

train_noise=''
if [ "${add_noise}" == "true" ]; then
    train_noise=${train_data}_aug
    echo "Add noise for train data"
fi
if [ ! -z $step02 ] && [ "${add_noise}" == "true" ] && [ "${trim_wav_scp}" == "true" ] ; then
  ${SCRIPT_ROOT}/common/noise/add-noise.sh --cmd "$cmd" --nj $nj --steps 1-5 --aug_list "$noise_list" \
        --subset-noise-rate $subset_noise_rate --sampling-rate $sampling_rate \
        --rir_dir $rir_dir --musan_noise_dir $musan_noise_dir \
        ${train_data} ${train_noise} || exit 1;
  echo "## LOG (step02): Add noise for train data done! "
fi

train_codecs=''
if [ "${do_codecs}" == "true" ]; then
    train_codecs=${train_data}_codecs
    echo "Add codecs for train data"
    if [ "${sampling_rate}" -ne "8000" ]; then
        echo "WARNNING: sampling_rate show be 8000" && exit 1;
    fi
fi

if [ ! -z $step03 ] && [ "${do_codecs}" == "true" ] && [ "${trim_wav_scp}" == "true" ] ; then
  tmpdir=${train_codecs}/tmp
  rm -r $tmpdir
  [ -d $tmpdir ] || mkdir -p $tmpdir
  if [ "${add_noise}" == "true" ];then
    utils/data/combine_data.sh $tmpdir/all ${train_data} ${train_noise} || exit 1;
  else
    utils/data/copy_data_dir.sh ${train_data} $tmpdir/all || exit 1;
  fi

  rm $tmpdir/all/reco2dur
  num_utt=$(wc -l $tmpdir/all/utt2spk | awk -v var=$subset_codec_rate '{print int($0*var);}' )
  utils/subset_data_dir.sh  $tmpdir/all $num_utt $tmpdir/sub || exit 1

  rm $tmpdir/sub/new_wav.scp $tmpdir/sub/new_segments $tmpdir/sub/reco2dur
  cp $tmpdir/sub/segments $tmpdir/sub/new_segments
  sed -e 's/^/codec-/'  $tmpdir/sub/wav.scp > $tmpdir/sub/new_wav.scp || exit 1
  cat $tmpdir/sub/new_segments |perl -ane 'use utf8; open qw(:std :utf8); chomp; m/(\S+)\s+(.*)/g or next; print "$1 codec-$2 $3 $4\n";' \
  > $tmpdir/sub/segments || exit 1

  cat $tmpdir/sub/new_wav.scp | \
  ${SCRIPT_ROOT}/common/codec/add-codec-with-ffmpeg.pl $sampling_rate $codec_list_file > $tmpdir/sub/wav.scp || exit 1;
  utils/fix_data_dir.sh $tmpdir/sub
  utils/data/copy_data_dir.sh  --utt-prefix "codec-" --spk-prefix "codec-" $tmpdir/sub ${train_codecs} || exit 1;

  rm -r $tmpdir
  echo "## LOG (step03): Add codecs for train data done! "
fi



#########################################################################################
##                                    mfcc feat extraction                             ##
#########################################################################################
if [ ! -z $step04 ]; then
  for sdata in $train_codecs $train_data $train_noise ; do
    data=$sdata/mfcc-hires feat=$sdata/feat/mfcc-hires/data log=$sdata/feat/mfcc-hires/log
    [ -d $data ] || mkdir -p $data
    utils/data/copy_data_dir.sh  $sdata $data || exit 1;
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
    --mfcc-config ${mfcc_hires_conf} $data $log $feat || exit 1;
    steps/compute_cmvn_stats.sh $data $log $feat || exit 1;
	  utils/fix_data_dir.sh $data
    echo "## LOG (step04): done with mfcc hires feat for training data '$data'"

	  data=$sdata/mfcc; feat=$sdata/feat/mfcc/data; log=$sdata/feat/mfcc/log
	  [ -d $data ] || mkdir -p $data
	  utils/data/copy_data_dir.sh  $sdata $data || exit 1;
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
    --mfcc-config ${mfcc_conf} $data $log $feat || exit 1;
    steps/compute_cmvn_stats.sh $data $log $feat || exit 1;
    utils/fix_data_dir.sh $data
    echo "## LOG (step04): done with mfcc feat for training data '$data'"
  done
  if [ "${add_noise}" == "true" ]; then
    utils/combine_data.sh ${train_data}_full/mfcc-hires $train_data/mfcc-hires $train_noise/mfcc-hires || exit 1;
    utils/combine_data.sh ${train_data}_full/mfcc $train_data/mfcc $train_noise/mfcc || exit 1;
    echo "Add noise for train data"
    if [ "${do_codecs}" == "true" ]; then
        utils/combine_data.sh ${train_data}_full/mfcc-hires $train_data/mfcc-hires $train_noise/mfcc-hires $train_codecs/mfcc-hires || exit 1;
        utils/combine_data.sh ${train_data}_full/mfcc $train_data/mfcc $train_noise/mfcc  $train_codecs/mfcc || exit 1;
        echo "Add noise and codecs for train data"
    fi
  else
    if [ "${do_codecs}" == "true" ]; then
        utils/combine_data.sh ${train_data}_full/mfcc-hires $train_data/mfcc-hires $train_codecs/mfcc-hires || exit 1;
        utils/combine_data.sh ${train_data}_full/mfcc $train_data/mfcc $train_codecs/mfcc || exit 1;
        echo "Add noise and codecs for train data"
    fi
  fi

  for sdata in $dev_man $dev_sge; do
    data=$sdata/mfcc-hires feat=$sdata/feat/mfcc-hires/data log=$sdata/feat/mfcc-hires/log
    [ -d $data ] || mkdir -p $data
    utils/data/copy_data_dir.sh  $sdata $data || exit 1;
    steps/make_mfcc.sh --cmd "$cmd" --nj 10 \
    --mfcc-config ${mfcc_hires_conf} $data $log $feat || exit 1;
    steps/compute_cmvn_stats.sh $data $log $feat || exit 1;
	  utils/fix_data_dir.sh $data
    echo "## LOG (step04): done with mfcc hires feat for testing data  '$data'"
  done

  echo "## LOG (step04): all feat extraction done!"
fi

#########################################################################################
##                              prepare dict and lang                                  ##
#########################################################################################
## dict
if [ ! -z $step05 ]; then
  [ -d $dictdir ] || mkdir -p $dictdir
  cp $src_dict/* $dictdir  || exit 1
  utils/validate_dict_dir.pl $dictdir || exit 1
  echo "## LOG (step05): make dict done with  '$dictdir' "
fi

## prepare lang
if [ ! -z $step06 ]; then
  [ -d $lang/tmp ] || mkdir -p $lang/tmp
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang || exit 1;
  echo "## LOG (step06): done with '$lang'"
fi

#########################################################################################
##                                  train GMM start                                    ##
#########################################################################################
##  train GMM default --steps 1-11  1.mono 2.ali 3. tri1 4.ali  5.tri2 6.ali  7.tri3 8.ali 9.lang-silprob 10.tri4 11.ali
if [ $train_gmm ]; then
  if [ ! -z $step07 ]; then
    echo "## LOG (step07): training started @ `date`"
    ${SCRIPT_ROOT}/common/gmm/run-gmm-v2.sh --cmd "$cmd" --nj $nj --steps $gmm_steps \
    --generate-ali-from-lats true --dict $dictdir \
    --train-id a --cmvn-opts "--norm-means=true"  --state-num $state_num --pdf-num $pdf_num \
    $train_data/mfcc $lang $expdir  || exit 1
    if [ "${add_noise}" == "true" ] || [ "${do_codecs}" == "true" ] ; then
      steps/copy_lat_dir.sh --nj $nj --cmd "$cmd" \
        --include-original true --prefixes "reverb1 babble music noise codec-reverb1 codec-babble codec-music codec-noise codec" \
        ${train_data}_full/mfcc $alidir ${alidir}_full || exit 1;
      steps/copy_ali_dir.sh --nj $nj --cmd "$cmd" \
        --include-original true --prefixes "reverb1 babble music noise codec-reverb1 codec-babble codec-music codec-noise codec" \
        ${train_data}_full/mfcc $alidir ${alidir}_full || exit 1;
    fi
    echo "## LOG (step07): done with '$expdir' @ `date`"
  fi
fi
##--------------------------------------    end    ------------------------------------##

if [ ! $train_gmm ]; then
  if [ ! -z $step08 ]; then
    echo "## LOG (step08) started alignment lattice @ `date`"
    steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj --generate-ali-from-lats true \
      $train_data/mfcc $lang $gsdir $alidir || exit 1;
    echo "## LOG (step08) ended alignment lattice @ `date`"
  fi
  if [ "${add_noise}" == "true" ] || [ "${do_codecs}" == "true" ] ; then
    steps/copy_lat_dir.sh --nj $nj --cmd "$cmd" \
        --include-original true --prefixes "reverb1 babble music noise codec-reverb1 codec-babble codec-music codec-noise codec" \
        ${train_data}_full/mfcc $alidir ${alidir}_full || exit 1;
    steps/copy_ali_dir.sh --nj $nj --cmd "$cmd" \
        --include-original true --prefixes "reverb1 babble music noise codec-reverb1 codec-babble codec-music codec-noise codec" \
        ${train_data}_full/mfcc $alidir ${alidir}_full || exit 1;
  fi
else
  dictdir=${dictdir}-silprob
  lang=${lang}-silprob
fi

if [ "${add_noise}" == "true" ]  || [ "${do_codecs}" == "true" ] ; then
    train_data=${train_data}_full
    alidir=${alidir}_full
    echo "Using add noise to train ivector and nnet3"
fi
#########################################################################################
##                                  train ivector start                                ##
#########################################################################################
if [ $train_ivector_extractor ]; then
  ## train lda mllt
  transform_dir=$nnetdir/pca_transform
  if [ ! -z $step09 ]; then
    steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
        --max-utts 100000 --subsample 2  \
      --splice-opts "--left-context=3 --right-context=3" \
      $train_data/mfcc-hires $transform_dir || exit 1
    echo "## LOG (step09): done with '$transform_dir' @ `date`"
  fi
  ## make diag ubm
  if [ ! -z $step10 ]; then
    steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj $nj \
    --num-threads 2 \
     --num-frames 720000 $train_data/mfcc-hires  512  $transform_dir $nnetdir/diag_ubm || exit 1
    echo "## LOG (step10): done with '$nnetdir/diag_ubm' @ `date`"
  fi
  ## train ivector
  if [ ! -z $step11 ]; then
    steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj $nj \
    --num-threads 1 --num-processes 1 \
    $train_data/mfcc-hires  $nnetdir/diag_ubm $ivector_extractor || exit 1;
    echo "## LOG (step11): done with '$ivector_extractor' @ `date`"
  fi
fi
##--------------------------------------    end    ------------------------------------##

#########################################################################################
##                                  extract train ivector                              ##
#########################################################################################
## ivector extraction
train_ivectors=$nnetdir/ivector-train
if [ ! -z $step12 ]; then
  data=$train_data/mfcc-hires
  data2=${data}-max2
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 $data  $data2
  echo "## LOG (step12): ivector extraction started @ `date` "
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" \
  --nj $nj $data2 $ivector_extractor $train_ivectors || exit 1;
  echo "## LOG (step12): ivector extraction done @ `date` "
fi

#########################################################################################
##                                  train tdnn start                                   ##
#########################################################################################
## train tdnn , if there something wrong change steps(default 1-7) train_stage(default -10) to skip succ step
## steps(default 1-7) 3.topo 4.tree 5.configs 6.train
if [ ! -z $step13 ]; then
  echo "## LOG (step13): tdnn training started @ `date`"
  ${SCRIPT_ROOT}/common/nnet3/train-tdnnf.sh --steps $tdnn_steps --cmd "$cmd --exclude=node01,node02" \
  --numleaves $numleaves  --hidden-dim ${hidden_dim} --chainname ${chainname}  \
  --bottleneck_dim $bottleneck_dim --small_dim $small_dim \
  --egs_opts "--frames-overlap-per-eg 0 --generate-egs-scp true" \
  --num_jobs_initial $num_jobs_initial --num_jobs_final $num_jobs_final \
  --train_stage $train_stage --train-ivectors $train_ivectors  \
  $train_data/mfcc  $train_data/mfcc-hires \
  $lang  $alidir $alidir  $nnetdir || exit 1;
  touch $nnetdir/${chainname}/egs/.nodelete || exit 1;
  echo "## LOG (step13): tdnn training done @ `date`"
fi

## prepare for language model
lmdir=$tgtdir/data/local/lm
lang_ngram=${lang}-${ngram}g
if [ ! -z $step14 ];then
  [ -d $lmdir ] || mkdir -p $lmdir
  awk '{$1=""; print;}' $train_data/mfcc-hires/text | \
  gzip -c > $lmdir/text.gz || exit 1;


  awk '{print $1;}' $dictdir/lexicon.txt \
  |  sort -u | gzip -c > $lmdir/vocab.gz || exit 1;

  ngram-count -order ${ngram} -wbdiscount -interpolate \
  -vocab $lmdir/vocab.gz  -unk -sort -text $lmdir/text.gz -lm $lmdir/lm${ngram}-wb.gz || exit 1;

  ${SCRIPT_ROOT}/common/lm/arpa2G.sh $lmdir/lm${ngram}-wb.gz $lang  $lang_ngram || exit 1;
  echo "## LOG (step14): done with '$lang_ngram'"
fi

## make graph
if [ ! -z $step15 ]; then
  echo "## LOG (step15): mkgraph started @ `date`"
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_ngram $nnetdir/${chainname} $graph || exit 1;
  echo "## LOG (step15): mkgraph done @ `date`"
fi

#########################################################################################
##                                  decode test set                                    ##
#########################################################################################
if [ ! -z $step16 ]; then
  echo "## LOG (step16): decoding started @ `date`"
  dev_mfcc_hires=$dev_man/mfcc-hires
  dev_name=dev-man-dominant
  dev_ivectors=$nnetdir/ivector-${dev_name}

  de_nj=50
  nx=$(wc -l < $dev_mfcc_hires/spk2utt)
  [ $nx -lt $de_nj ] && de_nj=$nx
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $de_nj $dev_mfcc_hires \
    $ivector_extractor $dev_ivectors || exit 1
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
  --nj $de_nj --cmd "$cmd" \
  --online-ivector-dir $dev_ivectors \
  $graph $dev_mfcc_hires $nnetdir/${chainname}/decode-${dev_name} || exit 1;
  echo "#$0 LOG Results......"
  ${SCRIPT_ROOT}/common/show-res.sh $nnetdir/${chainname}/decode-${dev_name}
  echo "## LOG (step16): done  @ `date`"
fi

if [ ! -z $step17 ]; then
  echo "## LOG (step17): decoding started @ `date`"
  dev_mfcc_hires=$dev_sge/mfcc-hires
  dev_name=dev-sge-dominant
  dev_ivectors=$nnetdir/ivector-${dev_name}

  de_nj=50
  nx=$(wc -l < $dev_mfcc_hires/spk2utt)
  [ $nx -lt $de_nj ] && de_nj=$nx
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj $de_nj $dev_mfcc_hires \
    $ivector_extractor $dev_ivectors || exit 1
  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
  --nj $de_nj --cmd "$cmd" \
  --online-ivector-dir $dev_ivectors \
  $graph $dev_mfcc_hires $nnetdir/${chainname}/decode-${dev_name} || exit 1;
  echo "#$0 LOG Results......"
  ${SCRIPT_ROOT}/common/show-res.sh $nnetdir/${chainname}/decode-${dev_name}
  echo "## LOG (step17): done  @ `date`"
fi
