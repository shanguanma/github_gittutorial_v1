#!/bin/bash

# v7 is same as v6, I try to removed spk0*(it represented the operator)
# 
. ./cmd.sh
. ./path.sh

set -e

steps=1
#train_plda_data_dir=/home/ellenrao/spkver/nist18/data/sre_combined
#nnet_dir=exp/xvector_nnet_1a/
num_components=1024 # the number of UBM components (used for VB resegmentation)
ivector_dim=400 # the dimension of i-vector (used for VB resegmentation)

version=v7
mfccdir=mfcc
vaddir=vad
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

# Now I make kaldi data(it contains wav.scp segments utt2spk spk2utt fullref.rttm labels reco2num_spk) 
#  from textGrid format transcription and wave file .
# step 1 make wav.scp
if [ ! -z $step01 ];then
   # unzip the file
   if [ ! -d data/test-data-20190808/sd_hdu_cn ];then
     tar xzf data/test-data-20190808/sd_hdu_cn.tar.gz -C data/test-data-20190808/
   fi
   internal_folder=data/test-data-20190808-tmp
   mkdir -p $internal_folder
   # the follow command can get abosulte path in ./wav-list.txt
   # find /home3/md510/w2019a/kaldi-recipe/speaker_diarization/data/test-data-20190808/sd_hdu_cn -name "*.wav" | sort > ./wav-list.txt
   find data/test-data-20190808/sd_hdu_cn -name "*.wav" | sort > $internal_folder/wav-list.txt
   internal_folder_small=data/test-data-20190808-tmp_entire_${version}
   mkdir -p $internal_folder_small
   head -n 300 $internal_folder/wav-list.txt > $internal_folder_small/wav-list.txt
   cat  $internal_folder_small/wav-list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";' | sort  > $internal_folder_small/wav.scp

fi
if [ ! -z $step02 ];then
  # add prefix (e.g:sd-hdu-cn) to recorder id 

  internal_folder_small=data/test-data-20190808-tmp_entire_${version}
  data=data/test-data-20190808-kaldi_entire_${version}
  mkdir -p $data
  cat $internal_folder_small/wav.scp | \
  source-md/egs/speaker_diarization/sge-manual-wavscp.pl sd-hdu-cn >  $data/wav.scp
  #source/egs/mandarin/update-lexicon-transcript-oct13/sge-manual-wavscp.pl sd-hdu-cn >  $data/wav.scp
fi


if [ ! -z $step03 ]; then
  internal_folder=data/test-data-20190808-tmp
  data=data/test-data-20190808-kaldi_entire_${version}
  if [ ! -e $internal_folder/text-grid-list.txt ]; then
    find  data/test-data-20190808/sd_hdu_cn -name "*.TextGrid" | sort -u > $internal_folder/text-grid-list.txt
  fi
  internal_folder_small=data/test-data-20190808-tmp_entire_${version}
  head -n 300 $internal_folder/text-grid-list.txt > $internal_folder_small/text-grid-list.txt
  cat $internal_folder_small/text-grid-list.txt | \
  source/egs/mandarin/update-lexicon-transcript-oct13/sge-manual-transcript.pl $data/wav.scp > $internal_folder_small/transcript.txt
  # remove silence duration,in other words, remove duration of no speaker info  
  # for example:$internal_folder_small/transcript.txt is as follow
  # sd-hdu-cn-0001-485995239007-spker-s001 sd-hdu-cn-0001-485995239007 0.00 39.96
  # sd-hdu-cn-0001-485995239007-spker-s001 sd-hdu-cn-0001-485995239007 39.96 40.32 S1m
  # .
  # .
  # .
  # we remove the line(e.g:sd-hdu-cn-0001-485995239007-non-non-spker-s001 sd-hdu-cn-0001-485995239007-non-non 0.00 39.96)
  awk '{if (NF==5){print $0}}' $internal_folder_small/transcript.txt>$internal_folder_small/transcript_new.txt
fi

# remove the operator(e.g:S0f, S0m) form $internal_folder_small/transcript_new.txt
if [ ! -z $step04 ];then
   # the following two commands are equivalent.
   # cat  $internal_folder_small/transcript_new.txt | grep -v 'S0f' | grep -v 'S0m'| awk '{print $1,$2,$4-$3,$5}' >  $internal_folder_small/transcript_new_remove_operator.txt 
   grep -v 'S0m\|S0f' $internal_folder_small/transcript_new.txt | awk '{print $1,$2,$4-$3,$5}' > $internal_folder_small/transcript_new_remove_operator.txt   
   # sort duration
   sort -k 3n $internal_folder_small/transcript_new_remove_operators.txt > $internal_folder_small/transcript_new_remove_operators_sort.txt

   # again remove  the operator (e.g:S0*, S10*) 
   grep -ivP '( s0| s10)' $internal_folder_small/transcript_new_remove_operators_sort.txt  > $internal_folder_small/transcript_new_thorough_remove_operators_sort.txt 
   # here getting shortest duration ,longest duration and average duration from remove operators transcript(e.g:$internal_folder_small/transcript_new_thorough_remove_operators_sort.txt)
   # get shortest duration
   head -n 1 $internal_folder_small/transcript_new_thorough_remove_operators_sort.txt > $internal_folder_small/transcript_new_thorough_remove_operators_sort_shortest.txt 
   # get longest duration
   # the following three command are equivalent.
   # sed -n '$p' $internal_folder_small/transcript_new_thorough_remove_operators_sort.txt > $internal_folder_small/transcript_new_thorough_remove_operators_sort_longest.txt
   # sed '$!D' $internal_folder_small/transcript_new_thorough_remove_operators_sort.txt > $internal_folder_small/transcript_new_thorough_remove_operators_sort_longest.txt
   awk 'END {print}' $internal_folder_small/transcript_new_thorough_remove_operators_sort.txt > $internal_folder_small/transcript_new_thorough_remove_operators_sort_longest.txt 
   # average duration
   awk '{sum +=$3}END{print sum/1666}' $internal_folder_small/transcript_new_thorough_remove_operators_sort.txt >  $internal_folder_small/transcript_new_thorough_remove_operators_sort_average.txt

   # get durations greater than  or equal to 1
   # here  
   grep -ivP '( s0| s10)' $internal_folder_small/transcript_new_remove_operators_sort.txt | awk '{ if($3 >=1){print;}  }'> $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub.txt   
   # get shortest duration
   head -n 1 $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub.txt > $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub_shortest.txt
   # get longest duration
   awk 'END {print}' $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub.txt> $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub_longest.txt
   # get average duration
   awk '{sum +=$3}END{print sum/666}' $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub.txt > $internal_folder_small/transcript_new_thorough_remove_operators_sort_sub_average.txt

   
fi 

