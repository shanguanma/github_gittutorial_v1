#!/bin/bash

#. path.sh
# make codec
# refrence: /home4/asr_resource/scripts/am/asr-training.sh  
#            /home3/zpz505/w2019/codec-augmented-from-ly/
# we use ffmpeg to downsample data to 8k HZ randomly using the codec provided in the codec list

# Refrence: "Audio codec simulation based data augmentation for telephony speech recognition"
#          address: https://ieeexplore.ieee.org/document/9023257

# first, use segment and rewrite-wav-scp.pl to rewrite wav.scp
# sconde, use  /home4/md510/package/source-md/asr_frontend/data_level/codec/get_utt2dur.sh to get utt2dur file
# third ,use utt2dur file to get new segement file
# fourth, use add-codec-with-ffmpeg.pl to rewrite wav.scp again.
# fifth, add prefix to segement




# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

SECONDS=0
stage=1
stop_stage=120
codec_list=/home4/md510/package/source-md/asr_frontend/data_level/codec/codec-list-full.txt
sampling_rate=8000
nj=10
src_data_dir= #data/train
tgt_data_dir= # data/train_codec
. path.sh
. cmd.sh
. utils/parse_options.sh

#if [ $# != 2 ]; then
#  echo "Usage: "
#  echo "  $0 [options] <srcdir> <destdir>"
#  echo "e.g.:"
#  echo " $0 --steps 1-2 --codec-list codec-list-full.txt -sampling-rate 8000 data/train data/train_codec"
#  echo "Options"
#  echo "   --stage                   # running stage"
#  echo "   --stop_stage              # stop stage"
#  echo "   --codec-list              # offer codec list, it must be set"
#  echo "   --sampling-rate           # downsample rate, it usually is 8000"
#
#fi



if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
  # 1. copy data to new folder 
  # this new folder is going to get codec data folder.
  utils/validate_data_dir.sh --no-feats  $src_data_dir
  [ -f $tgt_data_dir ] || rm -rf $tgt_data_dir
  utils/copy_data_dir.sh $src_data_dir $tgt_data_dir

fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
  log " Stage 2:  get trim wav.scp, new segments file"
   # check segments
  if [ ! -f ${tgt_data_dir}/segments ]; then
   log "## ERROR: segments file '${tgt_data_dir}/segments' not exist!"; 
  fi
  # check tmp_wav.scp
  if [ !  -f ${tgt_data_dir}/tmp_wav.scp ]; then
    log "## : tmp_wav.scp file exist! remove it and mv wav.sp to temp_wav.scp"
    rm -rf ${tgt_data_dir}/tmp_wav.scp
    mv  ${tgt_data_dir}/wav.scp ${tgt_data_dir}/tmp_wav.scp 
  fi
  # check tmp_segments
  if [ ! -f ${tgt_data_dir}/tmp_segments ]; then
   log "## : tmp_segments file exist!  remove it and segments to tmp_segments"
   rm -rf ${tgt_data_dir}/tmp_segments 
   mv  ${tgt_data_dir}/segments ${tgt_data_dir}/tmp_segments
  fi
  # make wav.scp
  cat ${tgt_data_dir}/tmp_segments | /home4/md510/package/source-md/asr_frontend/data_level/codec/rewrite-wav-scp.pl \
   $sampling_rate ${tgt_data_dir}/tmp_wav.scp > ${tgt_data_dir}/wav.scp|| exit 1;
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ];then
  log "Stage3: make segment ...."
  #make segments
  /home4/md510/package/source-md/asr_frontend/data_level/codec/get_utt2dur.sh --nj $nj --cmd "${train_cmd}" ${tgt_data_dir} || exit 1;
  cp ${tgt_data_dir}/utt2dur ${tgt_data_dir}/reco2dur
  cat ${tgt_data_dir}/utt2dur | \
    perl -ane 'use utf8; open qw(:std :utf8); chomp; m/(\S+)\s+(.*)/g or next; print "$1 $1 0.00 $2\n";' > ${tgt_data_dir}/segments
  utils/fix_data_dir.sh ${tgt_data_dir} || exit 1;
  # clean grabage file
  rm ${tgt_data_dir}/tmp_segments ${tgt_data_dir}/tmp_wav.scp
  log "Stage3 : rewrite reco2dur segments and wav.scp add trim done!"

fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
   log "Stage 4: modified wav.scp again"
   rm $tgt_data_dir/reco2dur
   utils/data/copy_data_dir.sh  $tgt_data_dir ${tgt_data_dir}/tmp
   # 3. modified wav.scp again
   sed -e 's/^/codec-/'  ${tgt_data_dir}/tmp/wav.scp > ${tgt_data_dir}/tmp/new_wav.scp || exit 1
   cat ${tgt_data_dir}/tmp/new_wav.scp | \
   /home4/md510/package/source-md/asr_frontend/data_level/codec/add-codec-with-ffmpeg.pl \
     $sampling_rate $codec_list > ${tgt_data_dir}/tmp/wav.scp || exit 1;

fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then
   log "Stage 5:  modified segments again"
   cp ${tgt_data_dir}/segments ${tgt_data_dir}/tmp/new_segments
   cat ${tgt_data_dir}/tmp/new_segments |perl -ane 'use utf8; open qw(:std :utf8); chomp; m/(\S+)\s+(.*)/g or next; print "$1 codec-$2 $3 $4\n";' \
    > ${tgt_data_dir}/tmp/segments || exit 1
   utils/fix_data_dir.sh  ${tgt_data_dir}/tmp
   utils/data/copy_data_dir.sh  --utt-prefix "codec-" --spk-prefix "codec-" ${tgt_data_dir}/tmp ${tgt_data_dir}
   
fi

log "Successfully finished. [elapsed=${SECONDS}s]"
