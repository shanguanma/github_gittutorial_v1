#!/bin/bash

# the script function is used to display lattice (the lattice is pruned from old decode lattice. we usualy call it 2-pass).
# how to run the script:
# current work folder:/home4/md510/w2019a/kaldi-recipe/add-noise-to-seame
# sbatch -o log/steps1-3.log source-md/egs/modify-decode-lattice/test-one-lattice.sh --steps 1-3 
. path.sh

# common option
cmd="slurm.pl --quiet"
steps=
nj=18

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

if [ ! -z $step01 ];then
  word_embedding='rnnlm-get-word-embedding exp/rnnlm/rnnlm_lstm_tdnn_1b/word_feats.txt exp/rnnlm/rnnlm_lstm_tdnn_1b/feat_embedding.final.mat -|'
  echo " generate one  pruned  lattice   "
  lattice-lmrescore-kaldi-rnnlm-pruned \
          --lm-scale=0.5 \
          --bos-symbol=34798 \
          --eos-symbol=34799 \
          --brk-symbol=34800 \
          --lattice-compose-beam=4 \
          --acoustic-scale=0.1 \
          --max-ngram-order=4 \
          data/lang_test_pp/G.fst \
          'rnnlm-get-word-embedding exp/rnnlm/rnnlm_lstm_tdnn_1b/word_feats.txt exp/rnnlm/rnnlm_lstm_tdnn_1b/feat_embedding.final.mat -|' \
          exp/rnnlm/rnnlm_lstm_tdnn_1b/final.raw \
          'ark:gunzip -c exp/chain_1a/tdnn_cnn_tdnnf_1a_sp/decode_dev_man_better_pp/lat.1.gz|' 'ark,t:|gzip -c>dev_man/two-pass/lat.1.gz'
fi

# we specify utterance to display lattice,so we require get utt-id from lattice.
if [ ! -z $step02 ];then
   echo " get utt-id list from lattice "
   gunzip -c  dev_man/two-pass/lat.1.gz | grep '^[a-z]' > dev_man/two-pass/utt-id-list.txt
   echo " random get one utt-id from utt-id list"
   sort --random-sort dev_man/two-pass/utt-id-list.txt | head -n 1 > dev_man/two-pass/one-utt-id.txt

fi
utt_id=$(cat dev_man/two-pass/one-utt-id.txt)
if [ ! -z $step03 ];then
   # convert one lattice to a normal fst
   # draw svg from a fst, because the pdf format don't display Chinese. output path is dev_man/two-pass
   local/show_lattice.sh \
           --output dev_man/two-pass \
           --format svg \
           --mode save \
           --lm-scale 0.5 \
           --acoustic-scale 0.1 \
           ${utt_id} dev_man/two-pass/lat.1.gz \
           data/lang_test_pp/words.txt

fi

