#!/bin/bash

. path.sh
. cmd.sh
stage=1

wer_detail_dir=/path/to/scoring_kaldi/wer_details    # it is obtain from local/score.sh


if [ $stage -le 1 ];then
   # I will going to count the deletion, insertion, subtitution errors,
   # then save the deletion.txt, insertion.txt, substitution.txt
   # there first column meaning is differnet , the rest is same.
   # the second column meaning is reference (it is real transcription, it is also ground truth)
   # the third column meaning is  hypothesis (it is recognition result)
   # the fourth column meaning is count.
   grep -P '^deletion' $wer_detail_dir/ops  > $wer_detail_dir/deletion.txt  
   grep -P '^insertion' $wer_detail_dir/ops  > $wer_detail_dir/insertion.txt
   grep -P '^substitution' $wer_detail_dir/ops  > $wer_detail_dir/substitution.txt  
fi
