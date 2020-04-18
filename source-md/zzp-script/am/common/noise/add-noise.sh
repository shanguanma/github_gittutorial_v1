#!/bin/bash

# Copyright 2018-2020 (Authors: zeng zhiping zengzp0912@gmail.com) 2020-03-18 updated
# This script built a nnet3 chain system,
# you can add noise, codecs in feat extraction steps
#


. path.sh
. cmd.sh

echo
echo "## LOG: $0 $@"
echo

# begin option
cmd="slurm.pl --quiet"
nj=50
steps=

aug_list="reverb music noise babble" #"reverb music noise babble clean"  #clean refers to the original train dir
sampling_rate=16000

# rate of all aug data 0<rate<1
subset_noise_rate=1

#reverberated speech
rir_dir=/home4/asr_resource/data/noise/RIRS_NOISES
num_reverb_copies=1

#add noise
musan_noise_dir=/home4/asr_resource/data/noise/musan/
# noise
noise_fg_interval=1
noise_bg_snrs=30:25:20:15:10:5 #15:10:5:0
# music
music_bg_snrs=30:25:20:15:10:5 #15:10:8:5
music_num_bg_noises=1 #1
# babble
babble_bg_snrs=30:25:20:15:10:5 #20:17:15:13
babble_num_bg_noises=1:2:3:4:5 #3:4:5:6:7

# end option

. parse_options.sh || exit 1

function Example {
 cat<<EOF
  sbatch -o /home/zpz505/w2019/seame/baseline/log/add-noise-step01-19.log \
 $0 --steps 1-19  \
  /home/zpz505/w2019/seame/baseline/data/train/mfcc-hires \
  /home/zpz505/w2019/seame/baseline-with-noise/data/train_aug



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

if [ $# -ne 2 ]; then
  Example && exit 1
fi

src_train=$1
tgt_train=$2

[ -d $tgt_train ] || mkdir -p $tgt_train

if [ ! -f $src_train/reco2dur ]; then
    utils/data/get_reco2dur.sh --nj $nj  --cmd "$cmd" $src_train || exit 1;
fi

# Make a version with reverberated speech
rvb_opts=()
rvb_opts+=(--rir-set-parameters "0.5, $rir_dir/simulated_rirs/smallroom/rir_list")
rvb_opts+=(--rir-set-parameters "0.5, $rir_dir/simulated_rirs/mediumroom/rir_list")
if [ ! -z $step01 ]; then
  # Make a reverberated version of the train.
  # Note that we don't add any additive noise here.
  steps/data/reverberate_data_dir.py \
    "${rvb_opts[@]}" \
    --speech-rvb-probability 1 \
    --prefix "reverb" \
    --pointsource-noise-addition-probability 0 \
    --isotropic-noise-addition-probability 0 \
    --num-replications $num_reverb_copies \
    --source-sampling-rate ${sampling_rate} \
    $src_train ${src_train}_reverb || exit 1
  echo "## LOG (step01): Make a reverberated version of the train '${src_train}_reverb' done!"
fi

if [ ! -z $step02 ]; then
  # Augment with musan_noise
  steps/data/augment_data_dir.py --utt-prefix "noise" --modify-spk-id "true" \
    --fg-interval $noise_fg_interval --fg-snrs "$noise_bg_snrs" --fg-noise-dir "$musan_noise_dir/noise" \
    $src_train ${src_train}_noise || exit 1
  echo "## LOG (step02): Make a noise version of the train '${src_train}_noise' done!"
fi

if [ ! -z $step03 ]; then
  # Augment with musan_music
  steps/data/augment_data_dir.py --utt-prefix "music" --modify-spk-id "true" \
    --bg-snrs "$music_bg_snrs" --num-bg-noises "$music_num_bg_noises" --bg-noise-dir "$musan_noise_dir/music" \
    $src_train ${src_train}_music || exit 1
  echo "## LOG (step02): Make a music version of the train '${src_train}_music' done!"
fi

if [ ! -z $step04 ]; then
  # Augment with musan_speech
  steps/data/augment_data_dir.py --utt-prefix "babble" --modify-spk-id "true" \
    --bg-snrs "$babble_bg_snrs" --num-bg-noises "$babble_num_bg_noises" --bg-noise-dir "$musan_noise_dir/speech" \
    $src_train ${src_train}_babble || exit 1
  echo "## LOG (step02): Make a babble version of the train '${src_train}_babble' done!"
fi

if [ ! -z $step05 ]; then
  # Combine all the augmentation dirs
  # This part can be simplified once we know what noise types we will add
  combine_str=""
  for n in $aug_list; do
    if [ "$n" == "clean" ]; then
      # clean refers to original of training directory
      combine_str+="$src_train "
    else
      combine_str+="${src_train}_${n} "
    fi
  done
  utils/combine_data.sh ${tgt_train}_full $combine_str || exit 1
  num_utt=$(wc -l ${tgt_train}_full/utt2spk | awk -v var=$subset_noise_rate '{print int($0*var);}' )
  utils/subset_data_dir.sh  ${tgt_train}_full $num_utt ${tgt_train} || exit 1
  rm -r ${tgt_train}_full ${src_train}_reverb ${src_train}_babble ${src_train}_music ${src_train}_noise || exit 1
  echo "## LOG (step02): Combine all version ('$aug_list') of the train '${tgt_train}' done!"
fi
