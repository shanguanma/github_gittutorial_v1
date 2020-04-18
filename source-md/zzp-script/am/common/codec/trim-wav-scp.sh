#!/bin/bash

# Copyright 2018-2020 (Authors: zeng zhiping zengzp0912@gmail.com) 2020-03-18 updated

. path.sh
. cmd.sh
#script_path='local-source'
[ ! -f ${SCRIPT_ROOT} ] || { echo "## ERROR ($0): please 'export SCRIPT_ROOT=/path/to/script' first! "; exit 1;  }

echo
echo "## LOG $0 $@"
echo

# begin option
cmd="slurm.pl --quiet"
steps=1
nj=40
# end option

function Example {
 cat <<EOF

 [Usage]:

 $0  [options] <data>
 # codes type: codec-list.txt codec-list-bad.txt codec-list-good.txt codec-list-medium.txt codec-list-full.txt
 [Example]:

 $0 --cmd "$cmd" 16000 ./train-data

EOF
}

. parse_options.sh || exit 1


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

sampling_rate=$1
train_data=$2

if [ ! -z $step01 ]; then
  # check segments
  [ -f ${train_data}/segments ] || \
  { echo "## ERROR ($0): segments file '${train_data}/segments' not exist!"; exit 1;  }
  # check tmp_wav.scp
  [ ! -f ${train_data}/tmp_wav.scp ] || { echo "## ERROR ($0): tmp_wav.scp file exist! "; exit 1;  }
  mv  ${train_data}/wav.scp ${train_data}/tmp_wav.scp || exit 1;
  # check tmp_segments
  [ ! -f ${train_data}/tmp_segments ] || { echo "## ERROR ($0): tmp_segments file exist! "; exit 1;  }
  mv  ${train_data}/segments ${train_data}/tmp_segments || exit 1;
  # make wav.scp
  cat ${train_data}/tmp_segments | ${SCRIPT_ROOT}/common/codec/rewrite-wav-scp.pl \
   $sampling_rate ${train_data}/tmp_wav.scp > ${train_data}/wav.scp|| exit 1;
  #make segments
  utils/data/get_utt2dur.sh --nj $nj ${train_data} || exit 1;
  cp ${train_data}/utt2dur ${train_data}/reco2dur
  cat ${train_data}/utt2dur | \
    perl -ane 'use utf8; open qw(:std :utf8); chomp; m/(\S+)\s+(.*)/g or next; print "$1 $1 0.00 $2\n";' > ${train_data}/segments
  utils/fix_data_dir.sh ${train_data} || exit 1;
  echo "## LOG : rewrite reco2dur segments and wav.scp add trim done!"
fi
