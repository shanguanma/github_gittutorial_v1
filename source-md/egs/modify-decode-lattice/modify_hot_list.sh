#!/bin/bash

. cmd.sh
. path.sh

# common option                                                                                       
steps=
nj=1                                                                                                
exp_root=output
nnet3_affix=_1a
input_dir=
# common option

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
#1. Prepared into kaldi format
# 1.1: make wav.scp
mkdir -p data/kaldi-format
if [ ! -z $step01 ];then
   find $input_dir -name "*.wav" > $input_dir/wav-list.txt
   cat  $input_dir/wav-list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";' |sort -u > data/kaldi-forma/wav.scp
fi
# 1.2: make utt2spk spk2utt, text
if [ ! -z $step02 ];then
   kaldi_format_dir=data/kaldi-format
   cat $kaldi_format_dir/wav.scp | awk '{print $1, $1;}' > $kaldi_format_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $kaldi_format_dir/utt2spk > $kaldi_format_dir/spk2utt

   cp -r $input_dir/text data/kaldi-format/
   # remove english comma, full stop. covert upper to lower
   cat data/kaldi-format/text |sed 's/[,.]//g'| sed 's/[A-Z]/[a-z]/g'> data/kaldi-format/text_1
   mv data/kaldi-format/text_1 data/kaldi-format/text

fi
# 2.make features
# 2.1: make 40 dimension mfcc features for
rubbish_dir=$exp_root/tmp
mkdir -p $rubbish_dir
if [ ! -z $step03 ];then
  
  mfccdir=$rubbish_dir/mfcc
  test_data=kaldi-format
  for dataset in $test_data; do
    steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj --mfcc-config conf/mfcc_hires.conf \
        data/${dataset} $rubbish_dir/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset} $rubbish_dir/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}  # remove segments with problems
  done

fi

# 2.2: make 100 dimension i-vector
# using zhiping's ivector-extractor
if [ ! -z $step04 ];then
  test_data=kaldi-format
  for dataset in $test_data; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj $nj \
      data/${dataset} /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/ivector-extractor \
      $rubbish_dir/nnet3${nnet3_affix}/ivectors_${dataset} || exit 1;
  done
fi

# 3. raw align decode (it also called one pass).
if [ ! -z $step05 ];then
   test_set=kaldi-format
   decode_nj=$nj
   dir=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/chain1024tdnnf
   graph_dir=$dir/graph_english_dict_with_mandarin_syllable
   for decode_set in $test_set; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "$train_cmd"  \
          --online-ivector-dir $rubbish_dir/nnet3${nnet3_affix}/ivectors_${dataset} \
          $graph_dir data/${decode_set} \
          $dir/decode_${decode_set} || exit 1;
  done
  raw_align_decode_dir=$exp_root/raw_alignd_decode
  mkdir -p $raw_align_decode_dir
  cp -r $dir/decode_${decode_set}/* $raw_align_decode_dir/
fi

# 4. oracle lattice 
# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step11 ];then
   # some check work
   test_data=kaldi-format
   data=data/$test_data
   # here $lang function  only offers words.txt
   #lang=asr_models_english_zhiping/lang-4glm-only-english
   lang=data/local/english_dict_with_mandarin_syllable_lang
   srcdir=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/chain1024tdnnf
   dir=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/chain1024tdnnf/decode_$test_data
   for f in $srcdir/{final.mdl,tree,cmvn_opts} $data/utt2spk $data/feats.scp \
     $lang/words.txt $lang/oov.txt; do
    if [ ! -f $f ]; then
     echo "$0: expected file $f to exist."
     exit 1
    fi
   done

   mkdir -p $dir
   cp $srcdir/final.mdl $dir
   cp $srcdir/tree $dir
   cp $srcdir/cmvn_opts $dir
   cp $srcdir/{splice_opts,delta_opts,final.mat,final.alimdl} $dir 2>/dev/null || true
   cp $srcdir/frame_subsampling_factor $dir 2>/dev/null || true

   if [ -f $srcdir/frame_subsampling_factor ]; then
    echo "$0: guessing that this is a chain system, checking parameters."
   fi

  frame_shift_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
  fi
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$train_cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir $dir/lattice_oracle
 
   # for easier management
  oracle_dir=$exp_root/oracle
  mkdir -p $oracle_dir 
  cp -r $dir/lattice_oracle/* $oracle_dir/
fi





