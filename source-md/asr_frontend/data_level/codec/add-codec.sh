#!/bin/bash


. path.sh


# make codec
# refrence: /home4/asr_resource/scripts/am/asr-training.sh  
#            /home3/zpz505/w2019/codec-augmented-from-ly/
# we use ffmpeg to downsample data to 8k HZ randomly using the codec provided in the codec list

steps=
codec_list=/home4/md510/package/source-md/asr_frontend/codec/codec-list-full.txt
sampling_rate=8000

. utils/parse_options.sh
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

if [ $# != 2 ]; then
  echo "Usage: "
  echo "  $0 [options] <srcdir> <destdir>"
  echo "e.g.:"
  echo " $0 --steps 1-2 --codec-list codec-list-full.txt -sampling-rate 8000 data/train data/train_codec"
  echo "Options"
  echo "   --steps                   # running stage"
  echo "   --codec-list              # offer codec list, it must be set"
  echo "   --sampling-rate           # downsample rate, it usually is 8000"
  
fi
src_add_data_dir=$1 #data/train
tgt_add_data_dir=$2

if [ ! -z $step01 ];then
  # 1. copy data to new folder 
  # this new folder is going to get codec data folder.
  utils/validate_data_dir.sh --no-feats  $src_add_data_dir
  [ -f $tgt_add_data_dir ] || rm -rf $tgt_add_data_dir
  utils/copy_data_dir.sh $src_add_data_dir $tgt_add_data_dir
  # 2. get trim wav.scp, new segments file
  source-md/asr_frontend/data_level/codec/trim-wav-scp.sh \
     $sampling_rate $tgt_add_data_dir

fi
if [ ! -z $step02 ];then
   rm $tgt_add_data_dir/reco2dur
   utils/data/copy_data_dir.sh  $tgt_add_data_dir ${tgt_add_data_dir}/tmp
   # 3. modified wav.scp again
   sed -e 's/^/codec-/'  ${tgt_add_data_dir}/tmp/wav.scp > ${tgt_add_data_dir}/tmp/new_wav.scp || exit 1
   cat ${tgt_add_data_dir}/tmp/new_wav.scp | \
   source-md/asr_frontend/data_level/codec/add-codec-with-ffmpeg.pl \
     $sampling_rate $codec_list > ${tgt_add_data_dir}/tmp/wav.scp || exit 1;
   # 4. modified segments again
   cp ${tgt_add_data_dir}/segments ${tgt_add_data_dir}/tmp/new_segments
   cat ${tgt_add_data_dir}/tmp/new_segments |perl -ane 'use utf8; open qw(:std :utf8); chomp; m/(\S+)\s+(.*)/g or next; print "$1 codec-$2 $3 $4\n";' \
    > ${tgt_add_data_dir}/tmp/segments || exit 1
   utils/fix_data_dir.sh  ${tgt_add_data_dir}/tmp
   utils/data/copy_data_dir.sh  --utt-prefix "codec-" --spk-prefix "codec-" ${tgt_add_data_dir}/tmp ${tgt_add_data_dir}
   
fi

