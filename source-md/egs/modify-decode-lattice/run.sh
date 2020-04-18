#!/bin/bash

. cmd.sh
. path.sh
set -e

echo
echo "## LOG: $0 $@"
echo
# current work folder:/home4/md510/w2019a/kaldi-recipe/chng_project_2019
# its means the script contains all path is relate the work folder if path isn't absolute Path.

# common option
#cmd="slurm.pl --quiet --nodelist=node07"
cmd="slurm.pl  --quiet --exclude=node05"
steps=
nj=18
#nj=50
exp_root=exp
nnet3_affix=_1a
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

test_set=test-wavdata-kaldi-format
mkdir -p data/$test_set
# prepare kaldi format from wav folder(e.g:)
# this script is test process if it is correct? so in wav folder .it only contains one wav file.
# make wav.scp
if [ ! -z $step01 ];then
   find /home4/md510/w2019a/kaldi-recipe/chng_project_2019/test-wavdata -name "*.wav" > /home4/md510/w2019a/kaldi-recipe/chng_project_2019/test-wavdata/wav-list.txt
   cat  /home4/md510/w2019a/kaldi-recipe/chng_project_2019/test-wavdata/wav-list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/wav.scp
   cat /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/wav.scp | sort -u >/home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/wav_1.scp
   mv /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/wav_1.scp /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/wav.scp
fi

if [ ! -z $step02 ];then
   #  make utt2spk spk2utt
   unseg_dir=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format 
   cat $unseg_dir/wav.scp | awk '{print $1, $1;}' > $unseg_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $unseg_dir/utt2spk > $unseg_dir/spk2utt

   cp -r /home4/md510/w2019a/kaldi-recipe/chng_project_2019/test-wavdata/text /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/
   # remove english comma, full stop.
   cat /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/text |sed 's/[,.]//g'> /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/text_1
   mv /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/text_1 /home4/md510/w2019a/kaldi-recipe/chng_project_2019/data/test-wavdata-kaldi-format/text
   
fi
if [ ! -z $step03 ];then
  mfccdir=`pwd`/mfcc
  vaddir=`pwd`/mfcc
  for name in test-wavdata-kaldi-format; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 3 \
      --cmd "$train_cmd" --write-utt2num-frames true \
      data/$name exp/make_mfcc $mfccdir
    utils/fix_data_dir.sh data/$name
  done

  for name in test-wavdata-kaldi-format; do
    sid/compute_vad_decision.sh --nj 3 --cmd "$train_cmd" \
      data/$name exp/make_vad $vaddir
    utils/fix_data_dir.sh data/$name
  done

  # The sre dataset is a subset of train
  #cp data/train/{feats,vad}.scp data/sre/
  #utils/fix_data_dir.sh data/sre

  # Create segments for test-wavdata-kaldi-format data.
  echo "0.01" > data/test-wavdata-kaldi-format/frame_shift
  diarization/vad_to_segments.sh --nj 3 --cmd "$train_cmd" \
    data/test-wavdata-kaldi-format data/test-wavdata-kaldi-format_segmented
  
fi

# cp -r data/test-wavdata-kaldi-format/text data/test-wavdata-kaldi-format_segmented/
# then manual cut text from subsegment and original text in data/test-wavdata-kaldi-format_segmented
# I don't know how to cut text utterance from subsegment auto.


# make 40 dim mfcc feature to be decoded.
if [ ! -z $step04 ];then
  test_set=test-wavdata-kaldi-format_segmented
  for dataset in $test_set; do
    # Create MFCCs for the eval set
    utils/copy_data_dir.sh data/$dataset data/${dataset}_hires
    steps/make_mfcc.sh --cmd "$train_cmd" --nj 3 --mfcc-config conf/mfcc_hires.conf \
        data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    steps/compute_cmvn_stats.sh data/${dataset}_hires $exp_root/make_hires/$dataset $mfccdir;
    utils/fix_data_dir.sh data/${dataset}_hires  # remove segments with problems
  done
 
fi
# step05-12 ,I used the model(e.g.path:) ly provided.
# ly's model path:/home/lyvt/work/projects/asr_models/SingaporeEnglish_0519NNET3
# model details are as follows:
# current folder:/home4/md510/w2019a/kaldi-recipe/chng_project_2019
# asr model:asr_models/SingaporeEnglish_0519NNET3/chain1024
# graph:asr_models/SingaporeEnglish_0519NNET3/chain1024/graph 
# i-vector-extrator:asr_models/SingaporeEnglish_0519NNET3/ivector-extractor
# lang_test:asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english
# note:/home/lyvt/work/projects/asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english/ is not complete.
#  it don't have phones folder and phones.txt,
# so i want to copy them from /home/lyvt/work/projects/asr_models/SingaporeEnglish_0519NNET3/chain1024/graph/phones,phones.txt 
# rnnlm model:asr_models/SingaporeEnglish_0519NNET3/rnnlm_tdnn_lstm_english
# make 100 dim ivector feture to be decoded
if [ ! -z $step05 ];then
  test_set=test-wavdata-kaldi-format_segmented
  for dataset in $test_set; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 3 \
      data/${dataset}_hires /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models/SingaporeEnglish_0519NNET3/ivector-extractor \
      $exp_root/nnet3${nnet3_affix}/ivectors_${dataset}_hires || exit 1;
  done

fi
# nnet3 decode,
# I used trained chain model to decode the test dataset.
if [ ! -z $step06 ];then
   test_set=test-wavdata-kaldi-format_segmented
   decode_nj=3
   dir=asr_models/SingaporeEnglish_0519NNET3/chain1024
   graph_dir=asr_models/SingaporeEnglish_0519NNET3/chain1024/graph
   for decode_set in $test_set; do 
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "${cmd}"  \
          --online-ivector-dir exp/nnet3_1a/ivectors_${decode_set}_hires \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set} || exit 1;
  done
fi
# result: vim asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented/scoring_kaldi/best_wer
# %WER 45.20 

# online decode, I don't run it.
if [ ! -z $step100 ];then
  # note: if the features change (e.g. you add pitch features), you will have to
  # change the options of the following command line.
  steps/online/nnet3/prepare_online_decoding.sh \
       --mfcc-config conf/mfcc_hires.conf \
       $lang exp/nnet3/extractor $dir ${dir}_online

  rm $dir/.error 2>/dev/null || true
  for decode_set in train_dev eval2000 $maybe_rt03; do
    (
      # note: we just give it "$decode_set" as it only uses the wav.scp, the
      # feature type does not matter.

      steps/online/nnet3/decode.sh --nj $decode_nj --cmd "$decode_cmd" \
          --acwt 1.0 --post-decode-acwt 10.0 \
         $graph_dir data/${decode_set}_hires \
         ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_tg || exit 1;
      if $has_fisher; then
          steps/lmrescore_const_arpa.sh --cmd "$decode_cmd" \
            data/lang_sw1_{tg,fsh_fg} data/${decode_set}_hires \
            ${dir}_online/decode_${decode_set}${decode_iter:+_$decode_iter}_sw1_{tg,fsh_fg} || exit 1;
      fi
    ) || touch $dir/.error &
  done
  wait
  if [ -f $dir/.error ]; then
    echo "$0: something went wrong in decoding"
    exit 1
  fi
fi

# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step07 ];then
   # some check work
   data=data/test-wavdata-kaldi-format_segmented_hires
   # here $lang function  only offers words.txt
   lang=asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english
   srcdir=asr_models/SingaporeEnglish_0519NNET3/chain1024
   dir=asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented
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

  #utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  #cp $lang/phones.txt $dir

  frame_shift_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
  fi
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir $dir/lattice_oracle
fi
# result:oracle %WER is: 32.03%

# convert oracle lattice to fst and display.
# I get know that oracle lattice is single path fst.
if [ ! -z $step08 ];then
  # note: if lm_scale=0.0,acoustic_scale=0.0 ,get fst has not weigth
  #       lm_scale=0.5, acoustic_scale=0.1 is learned from rnnlm/lmrescore_pruned.sh 
  decode_name=decode_test-wavdata-kaldi-format_segmented
  uttid=audio1-1-00000-01779
  lat=asr_models/SingaporeEnglish_0519NNET3/chain1024/${decode_name}/lattice_oracle
  words=asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english/words.txt
  local/show_lattice.sh \
        --output $lat \
        --format svg --mode save --lm-scale 0.5 \
        --acoustic-scale 0.1 $uttid $lat/lat.1.gz $words
fi
# display oracle lattice by text format
if [ ! -z $step09 ];then
   latdir=asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented/lattice_oracle/lat.1.gz
   echo "display oracle lattice by text format ...."
   lattice-copy "ark:gunzip -c $latdir |" ark,t:- | utils/int2sym.pl -f 3 asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english/words.txt | head -n 100   
fi

# get nbest path from chain model decode lattice, in other words, I use $dir/decode_${decode_set}_better_pp/lat.*.gz in step4 of the script.
# refrence:/home3/md510/kaldi/egs/callhome_egyptian/s5/local/lattice_main.sh
#         :/home3/md510/kaldi/egs/callhome_egyptian/s5/local/get_nbest.sh 
#         :rnnlm/lmrescore_nbest.sh
# First convert lattice to N-best.  Be careful because this
# will be quite sensitive to the acoustic scale; this should be close
# to the one we'll finally get the best WERs with.
# Note: the lattice-rmali part here is just because we don't
# need the alignments for what we're doing.
# the step isn't run .
if [ ! -z $step10 ]; then
  keep_ali=true
  acwt=0.1 # from rnnlm/lmrescore_nbest.sh
  N=10
  indir=exp/chain_1a/tdnn_cnn_tdnnf_1a_sp/decode_test-wavdata-kaldi-format_better_pp
  dir=exp/chain_1a/tdnn_cnn_tdnnf_1a_sp/decode_test-wavdata-kaldi-format_better_pp_nbest
  nj=3 # remember to revisit. 
  mkdir -p $dir
  echo "$0: converting lattices to N-best lists."
  if $keep_ali; then
    $cmd JOB=1:$nj $dir/log/lat2nbest.JOB.log \
      lattice-to-nbest --acoustic-scale=$acwt --n=$N \
      "ark:gunzip -c $indir/lat.JOB.gz|" \
      "ark:|gzip -c >$dir/nbest1.JOB.gz" || exit 1;
  else
    $cmd JOB=1:$nj $dir/log/lat2nbest.JOB.log \
      lattice-to-nbest --acoustic-scale=$acwt --n=$N \
      "ark:gunzip -c $indir/lat.JOB.gz|" ark:- \|  \
      lattice-rmali ark:- "ark:|gzip -c >$dir/nbest1.JOB.gz" || exit 1;
  fi
fi
 

# rnnlm lattice rescore decode lattice
if [ ! -z $step11 ];then
   echo "$0: Perform lattice-rescoring on $indir"
    #${root_dir}/data/lang-4g  is old lm 
    #$dir is trained already new lm
    #${root_dir}/data/${decode_set}_hires is test data path
    #decode_dir is old test decode path,dnn decode
    test_set=test-wavdata-kaldi-format_segmented
    ngram_order=4
    test_lang=asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english
    rnnlmdir=asr_models/SingaporeEnglish_0519NNET3/rnnlm_tdnn_lstm_english 
      for decode_set in ${test_set}; do
       indir=asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented    #  old score folder
       dev_rescore_dir=asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_${decode_set}_lattice_rescore  #  rescore output folder
      #lattice rescoring
        rnnlm/lmrescore_pruned.sh \
         --cmd "$cmd --mem 4G" \
         --weight 0.5 --max-ngram-order $ngram_order \
         ${test_lang} $rnnlmdir \
         data/${decode_set}_hires  $indir \
         $dev_rescore_dir
     done
# result:asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented_lattice_rescore/scoring_kaldi/best_wer
# WER 45.91
fi
# rnnlm nbest rescore decode lattice
if [ ! -z $step12 ]; then
   echo "$0: Perform nbest-rescoring on $indir"
   test_set=test-wavdata-kaldi-format_segmented
   test_lang=asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english
   rnnlmdir=asr_models/SingaporeEnglish_0519NNET3/rnnlm_tdnn_lstm_english
   for decode_set in ${test_set}; do
     indir=asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented     #old score folder
      dev_rescore_dir=asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_${decode_set}_nbest_rescore  # dev_man and dev_sge rescore output folder.
     
     # nbest rescoring
     rnnlm/lmrescore_nbest.sh \
        --cmd "$cmd --mem 4G" --N 20 \
         0.8 ${test_lang} $rnnlmdir \
         data/${decode_set}_hires   $indir \
         $dev_rescore_dir
    done
# result:asr_models/SingaporeEnglish_0519NNET3/chain1024/decode_test-wavdata-kaldi-format_segmented_nbest_rescore/scoring_kaldi/best_wer
# WER 45.55
fi


# step13- ,I used the model(e.g.path:) zhiping provided.
# model origianl path:
# Acoustic model:        /home3/zpz505/w2019/code-switch/update-lexicon-transcript-feb/exp/tdnn/chain1024tdnnf
# Rnn LM:               /home3/zpz505/w2019/code-switch/update-lexicon-transcript-mar/exp/rnnlm_tdnn_lstm_english
# English graph:        /home3/zpz505/w2019/code-switch/update-lexicon-transcript-feb/exp/tdnn/chain1024tdnnf/graph-4gram-only-english
# English lang:        /home3/zpz505/w2019/code-switch/update-lexicon-transcript-feb/data/lang-only-english
# English dict:        /home3/zpz505/w2019/code-switch/update-lexicon-transcript-feb/data/local/dict-only-english
# dict:               /home4/hhx502/w2019/exp/dec-02-2018-telium-data/exp/ivector-extractor
# lang_test:asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english
# model details are as follows:
# current folder:/home4/md510/w2019a/kaldi-recipe/chng_project_2019
# asr model:asr_models_english_zhiping/chain1024tdnnf
# i-vector-extrator:asr_models_english_zhiping/ivector-extractor
# lang_test(contains G.fst):asr_models/SingaporeEnglish_0519NNET3/lang-4glm-only-english
# lang:asr_models_english_zhiping/lang-only-english
# rnnlm model:asr_models_english_zhiping/rnnlm_tdnn_lstm_english
# make 100 dim ivector feture to be decoded
if [ ! -z $step13 ];then
  test_set=test-wavdata-kaldi-format_segmented
  for dataset in $test_set; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj 3 \
      data/${dataset}_hires /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/ivector-extractor \
      $exp_root/nnet3${nnet3_affix}/ivectors_${dataset}_hires_zhiping || exit 1;
  done

fi
# nnet3 decode,
# I used trained chain model to decode the test dataset.
if [ ! -z $step14 ];then
   test_set=test-wavdata-kaldi-format_segmented
   decode_nj=3
   dir=asr_models_english_zhiping/chain1024tdnnf
   graph_dir=asr_models_english_zhiping/chain1024tdnnf/graph-4gram-only-english
   for decode_set in $test_set; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $decode_nj --cmd "${cmd}"  \
          --online-ivector-dir exp/nnet3_1a/ivectors_${decode_set}_hires_zhiping \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set} || exit 1;
  done
fi
# result: vim asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented/scoring_kaldi/best_wer 
# %WER 44.41 [ 131 / 295, 25 ins, 28 del, 78 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented/wer_12_0.0 



# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step15 ];then
   # some check work
   data=data/test-wavdata-kaldi-format_segmented_hires
   # here $lang function  only offers words.txt
   #lang=asr_models_english_zhiping/lang-4glm-only-english
   lang=asr_models_english_zhiping/lang-only-english
   srcdir=asr_models_english_zhiping/chain1024tdnnf
   dir=asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented
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

  #utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  #cp $lang/phones.txt $dir

  frame_shift_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
  fi
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir $dir/lattice_oracle
fi
# result:cat log/steps_new_13-17_1.log 
# overall oracle %WER is: 22.71%


# rnnlm lattice rescore decode lattice
if [ ! -z $step16 ];then
   echo "$0: Perform lattice-rescoring on $indir"
    #${root_dir}/data/lang-4g  is old lm 
    #$dir is trained already new lm
    #${root_dir}/data/${decode_set}_hires is test data path
    #decode_dir is old test decode path,dnn decode
    test_set=test-wavdata-kaldi-format_segmented
    ngram_order=4
    test_lang=asr_models_english_zhiping/lang-4glm-only-english
    rnnlmdir=asr_models_english_zhiping/rnnlm_tdnn_lstm_english
      for decode_set in ${test_set}; do
       indir=asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented    #  old score folder
       dev_rescore_dir=asr_models_english_zhiping/chain1024tdnnf/decode_${decode_set}_lattice_rescore  #  rescore output folder
      #lattice rescoring
        rnnlm/lmrescore_pruned.sh \
         --cmd "$cmd --mem 4G" \
         --weight 0.5 --max-ngram-order $ngram_order \
         ${test_lang} $rnnlmdir \
         data/${decode_set}_hires  $indir \
         $dev_rescore_dir
     done
#%WER 40.00 [ 118 / 295, 23 ins, 26 del, 69 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_lattice_rescore/wer_13_0.5
fi
# rnnlm nbest rescore decode lattice
if [ ! -z $step17 ]; then
   echo "$0: Perform nbest-rescoring on $indir"
   test_set=test-wavdata-kaldi-format_segmented
   test_lang=asr_models_english_zhiping/lang-4glm-only-english
   rnnlmdir=asr_models_english_zhiping/rnnlm_tdnn_lstm_english
   for decode_set in ${test_set}; do
     indir=asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented     #old score folder
      dev_rescore_dir=asr_models_english_zhiping/chain1024tdnnf/decode_${decode_set}_nbest_rescore  # dev_man and dev_sge rescore output folder.

     # nbest rescoring
     rnnlm/lmrescore_nbest.sh \
        --cmd "$cmd --mem 4G" --N 20 \
         0.8 ${test_lang} $rnnlmdir \
         data/${decode_set}_hires   $indir \
         $dev_rescore_dir
    done
# %WER 42.71 [ 126 / 295, 26 ins, 23 del, 77 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_nbest_rescore/wer_13_0.0
fi

# I want to use specify some new words to extend G.fst to imporve asr WER.
# reference: /home3/md510/kaldi/egs/mini_librispeech/s5/local/grammar/extend_vocab_demo.sh
tree_dir=exp/chain/tree_sp # make set the path by yourself.
lang_base=data/lang_nosp_basevocab
lang_ext=data/lang_nosp_extvocab
dict=asr_models_english_zhiping/dict-only-english
run_g2p=true
if [ ! -z $step18 ]; then
  mkdir -p data/local/dict_nosp_basevocab
  cp -r $dict/* data/local/dict_nosp_basevocab
  echo "#nonterm:unk" > data/local/dict_nosp_basevocab/nonterminals.txt

  utils/prepare_lang.sh data/local/dict_nosp_basevocab \
       "<unk>" data/local/lang_tmp_nosp $lang_base
fi

if [ ! -z $step19 ]; then
  # note: <unk> does appear in that arpa file, with a reasonable probability
  # (0.0)...  presumably because the vocab that the arpa file was built with was
  # not vast, so there were plenty of OOVs.  It would be possible to adjust its
  # probability with adjust_unk_arpa.pl, but for now we just leave it as-is.
  # The <UNK> appears quite a few times in the ARPA.  In the language model we
  # replaced it with #nonterm:unk, which will later expand to our custom graph
  # of new words.

  # We don't want the #nonterm:unk on the output side of G.fst, or it would
  # appear in the decoded output, so we remove it using the 'fstrmsymbols' command.

  nonterm_unk=$(grep '#nonterm:unk' $lang_base/words.txt | awk '{print $2}')

 # gunzip -c  data/local/lm/lm_tgsmall.arpa.gz | \
    #sed 's/<unk>/#nonterm:unk/g' | \
    #arpa2fst --disambig-symbol=#0 \
    #         --read-symbol-table=$lang_base/words.txt - |
    fstrmsymbols --remove-from-output=true "echo $nonterm_unk|" asr_models_english_zhiping/lang-4glm-only-english/G.fst $lang_base/G.fst
fi

if [ ! -z $step20 ]; then
  # make the top-level part of the graph.
  # make copy some important  files from am model
  cp -r  asr_models_english_zhiping/chain1024tdnnf/cmvn_opts   exp/chain/tree_sp/
  cp -r  asr_models_english_zhiping/chain1024tdnnf/final.*   exp/chain/tree_sp/
  cp -r  asr_models_english_zhiping/chain1024tdnnf/frame_subsampling_factor   exp/chain/tree_sp/
  cp -r  asr_models_english_zhiping/chain1024tdnnf/phones.txt   exp/chain/tree_sp/
  cp -r  asr_models_english_zhiping/chain1024tdnnf/srand   exp/chain/tree_sp/
  cp -r  asr_models_english_zhiping/chain1024tdnnf/tree   exp/chain/tree_sp/
 
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_base $tree_dir $tree_dir/extvocab_nosp_top
fi

if [ ! -z $step21 ] && $run_g2p; then
  # you may have to do some stuff manually to install sequitur, to get this to work.
  dict=data/local/dict_nosp_basevocab
  steps/dict/train_g2p.sh --silence-phones $dict/silence_phones.txt $dict/lexicon.txt  $tree_dir/extvocab_nosp_g2p
fi


if [ ! -z $step22 ]; then
  # Create data/local/dict_nosp_newvocab as a dict-dir containing just the
  # newly created vocabulary entries (but the same phone list as our old setup, not
  # that it matters)

  mkdir -p $tree_dir/extvocab_nosp_lexicon

  # First find a list of words in the test set that are out of vocabulary.
  # Of course this is totally cheating.
  awk -v w=asr_models_english_zhiping/lang-only-english/words.txt 'BEGIN{while(getline <w) seen[$1] = $1} {for(n=2;n<=NF;n++) if(!($n in seen)) oov[$n] = 1}
                                END{ for(k in oov) print k;}' < data/test-wavdata-kaldi-format_segmented_hires/text > $tree_dir/extvocab_nosp_lexicon/words
  echo "$0: generating g2p entries for $(wc -l <$tree_dir/extvocab_nosp_lexicon/words) words"

  if $run_g2p; then
    steps/dict/apply_g2p.sh $tree_dir/extvocab_nosp_lexicon/words $tree_dir/extvocab_nosp_g2p  $tree_dir/extvocab_nosp_lexicon
  fi

  # extend_lang.sh needs it to have basename 'lexiconp.txt'.
  mv $tree_dir/extvocab_nosp_lexicon/lexicon.lex $tree_dir/extvocab_nosp_lexicon/lexiconp.txt

  [ -f data/lang_nosp_extvocab/G.fst ] && rm data/lang_nosp_extvocab/G.fst
  utils/lang/extend_lang.sh  data/lang_nosp_basevocab $tree_dir/extvocab_nosp_lexicon/lexiconp.txt  data/lang_nosp_extvocab
fi

if [ ! -z $step23 ]; then
  # make the G.fst for the extra words.  Just assign equal probabilities to all of
  # them.  The words will all transition from state 1 to 2.
  cat <<EOF > $lang_ext/G.txt
0    1    #nonterm_begin <eps>
2    3    #nonterm_end <eps>
3
EOF
  lexicon=$tree_dir/extvocab_nosp_lexicon/lexiconp.txt
  num_words=$(wc -l <$lexicon)
  cost=$(perl -e "print log($num_words)");
  awk -v cost=$cost '{print 1, 2, $1, $1, cost}' <$lexicon >>$lang_ext/G.txt
  fstcompile --isymbols=$lang_ext/words.txt --osymbols=$lang_ext/words.txt <$lang_ext/G.txt | \
    fstarcsort --sort_type=ilabel >$lang_ext/G.fst
fi

if [ ! -z $step24 ]; then
  # make the part of the graph that will be included.
  # Refer to the 'compile-graph' commands in ./simple_demo.sh for how you'd do
  # this in code.
  utils/mkgraph.sh --self-loop-scale 1.0 $lang_ext $tree_dir $tree_dir/extvocab_nosp_part
fi

if [ ! -z $step25 ]; then
  offset=$(grep nonterm_bos $lang_ext/phones.txt | awk '{print $2}')
  nonterm_unk=$(grep nonterm:unk $lang_ext/phones.txt | awk '{print $2}')

  mkdir -p $tree_dir/extvocab_nosp_combined
  [ -d $tree_dir/extvocab_nosp_combined/phones ] && rm -r $tree_dir/extvocab_nosp_combined/phones
  # the decoding script expects words.txt and phones/, copy them from the extvocab_part
  # graph directory where they will have suitable values.
  cp -r $tree_dir/extvocab_nosp_part/{words.txt,phones.txt,phones/} $tree_dir/extvocab_nosp_combined

  # the following, due to --write-as-grammar=false, compiles it into an FST
  # which can be decoded by our normal decoder.
  make-grammar-fst --write-as-grammar=false --nonterm-phones-offset=$offset $tree_dir/extvocab_nosp_top/HCLG.fst \
                   $nonterm_unk $tree_dir/extvocab_nosp_part/HCLG.fst  $tree_dir/extvocab_nosp_combined/HCLG.fst

  # the following compiles it and writes as GrammarFst.  The size is 2.2G, vs. 2.2G for HCLG.fst.
  # In other examples, of course the difference might be more.

  make-grammar-fst --write-as-grammar=true --nonterm-phones-offset=$offset $tree_dir/extvocab_nosp_top/HCLG.fst \
                $nonterm_unk $tree_dir/extvocab_nosp_part/HCLG.fst  $tree_dir/extvocab_nosp_combined/HCLG.gra
fi


if [ ! -z $step26 ]; then
  # We just replace the graph with the one in $treedir/extvocab_nosp_combined.

  steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 3 \
    --cmd "$train_cmd" --online-ivector-dir exp/nnet3_1a/ivectors_test-wavdata-kaldi-format_segmented_hires_zhiping \
    exp/chain/tree_sp/extvocab_nosp_combined data/test-wavdata-kaldi-format_segmented_hires asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_ev_nosp_comb
 # %WER 44.90 [ 132 / 294, 24 ins, 30 del, 78 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_ev_nosp_comb/wer_12_0.5
fi

if [ ! -z $step27 ];then
  steps/nnet3/decode_grammar.sh --acwt 1.0 --post-decode-acwt 10.0 --frames-per-chunk 140 --nj 3 \
    --cmd "$train_cmd" --online-ivector-dir exp/nnet3_1a/ivectors_test-wavdata-kaldi-format_segmented_hires_zhiping \
    exp/chain/tree_sp/extvocab_nosp_combined data/test-wavdata-kaldi-format_segmented_hires asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_ev_nosp_comb_gra

  #  The WER when decoding with the grammar FST directly is exactly the same:
  # %WER 44.90 [ 132 / 294, 24 ins, 30 del, 78 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_ev_nosp_comb_gra/wer_12_0.5
fi


# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step28 ];then
   # some check work
   data=data/test-wavdata-kaldi-format_segmented_hires
   # here $lang function  only offers words.txt
   #lang=asr_models_english_zhiping/lang-4glm-only-english
   lang=asr_models_english_zhiping/lang-only-english
   srcdir=asr_models_english_zhiping/chain1024tdnnf
   dir=asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_ev_nosp_comb_gra
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

  #utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  #cp $lang/phones.txt $dir

  frame_shift_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
  fi
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir $dir/lattice_oracle
# results:cat log/steps_new_28.log
# overall oracle %WER is: 23.81%
# oracle hyp: vim asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_ev_nosp_comb_gra/lattice_oracle/oracle_hyp.txt
fi

# use keyword method to improve asr WER.
# refrence: /home2/tungpham/hotlist_rescoring_MaduoSys/script_index_search/modifyLattice_biasKeywords.sh
# current folder path:/home4/md510/w2019a/kaldi-recipe/chng_project_2019

# Generating key word index lattice requires the follows files as input files. 
lang_dir=asr_models_english_zhiping/lang-4glm-only-english # the lang contains G.fst 
model=asr_models_english_zhiping/chain1024tdnnf/final.mdl # acoustic model
# the decode dir is one pass decode results.one pass means that it does not make any rescores.
decodedir=asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented
data_dir=data/test-wavdata-kaldi-format_segmented_hires # test data dir ,it is 40 dimension data folder.

# Generating key word index lattice requires the follows configur file
acwt=0.076923077 # i.e. 1/13: acoustic scale 
lmwt=1.0 #LM scale


keyword_method_root_dir=exp/keyword_tung
indices=$keyword_method_root_dir/index_data/dev
wbound=${lang_dir}/phones/word_boundary.int

kwsdatadir=$keyword_method_root_dir/datadir
[ -d $kwsdatadir ] || mkdir -p $kwsdatadir
[ -d $indices ] || mkdir -p $indices
if [ ! -z $step29 ];then

  cp -r $datadir/segments $kwsdatadir/segments
  utter_id=$kwsdatadir/utter_id
  awk '{print $1,NR}' exp/keyword_tung/datadir/segments > $utter_id
 
  log=$indices/log
  [ -d $log ] ||mkdir -p $log
  nj=`cat $decodedir/num_jobs`

  $cmd JOB=1:$nj $indices/log/index.JOB.log \
  lattice-align-words $wbound  $model "ark:gzip -cdf $decodedir/lat.JOB.gz|" ark:- \| \
  lattice-scale --acoustic-scale=$acwt --lm-scale=$lmwt ark:- ark:- \| \
  lattice-to-kws-index ark:$utter_id ark:- ark:- \| \
  kws-index-union --skip-optimization=true ark:- "ark:|gzip -c > $indices/index.JOB.gz"
  echo "make key word index lattice"
fi

# make keywords fst.
# example is as /home2/tungpham/hotlist_rescoring_MaduoSys/script_index_search/keywords.txt
# make it by yourself. These are the so-called keywords or hot words you define.
keywordList=/home2/tungpham/hotlist_rescoring_MaduoSys/script_index_search/keywords.txt 

if [ ! -z $step30 ];then
  cp -r $decodedir/num_jobs $indices/num_jobs
  nj=`cat $indices/num_jobs`

  #Create FST keywords. Only IV keywords can be converted to fst 
  #(in $kwsdatadir/keywords.fsts), OOV keywords are in $kwsdatadir/oov_kwid.txt
  keyword_text=$kwsdatadir/keywords.txt # input files
  keyword_int=$kwsdatadir/keywords.int  # output files
  keyword_fsts=$kwsdatadir/keywords.fsts # output files
  cp $keywordList $keyword_text
  perl /home2/tungpham/hotlist_rescoring_MaduoSys/script_index_search/make_int_kwlist.pl \
       ${lang_dir}/words.txt <$keyword_text >$keyword_int 2> $kwsdatadir/oov_kwid.txt
  echo -e "\n OOV keyword list in $kwsdatadir/oov_kwid.txt \n";
  transcripts-to-fsts ark:$keyword_int ark:$keyword_fsts

fi


kwsoutput=$keyword_method_root_dir/search_results/dev
mkdir -p $kwsoutput
### search 
if [ ! -z $step31 ];then
  /home2/tungpham/hotlist_rescoring_MaduoSys/steps/search_index.sh \
    --cmd "$train_cmd" --indices-dir $indices \
    $kwsdatadir $kwsoutput keywords.fsts  || exit 1

fi


alignLat=$keyword_method_root_dir/new_lats/alignLat
orgLatMod=$keyword_method_root_dir/new_lats/orgLatMod
alignLatMod=$keyword_method_root_dir/new_lats/alignLatMod
tmp=$keyword_method_root_dir/tmp
mkdir -p $alignLat
mkdir -p $orgLatMod
mkdir -p $alignLatMod
mkdir -p $tmp
#nj=`cat $indices/num_jobs`
if [ ! -z $step32 ];then
  #for (( i = 1 ; i <= $nj ; i++ ));do
 #source /home2/tungpham/hotlist_rescoring_MaduoSys/script_index_search/path.sh
 nj=`cat $indices/num_jobs`
 for i in 1 2 3;do  
  ##Modify lattice
  echo "start modify lattice"
  #$train_cmd i=1:$nj \
  kaldi_lattice_invest_ver \
    --acoustic-scale=$acwt $model "ark:gzip -cdf $decodedir/lat.$i.gz|" \
   $wbound $kwsoutput/result.$i $kwsdatadir/keywords.int \
   $tmp/lat_trav.$i $tmp/lat_trav_log.$i.txt \
   "ark:|gzip -c >$alignLat/lat.$i.gz" "ark:|gzip -c >$orgLatMod/lat.$i.gz" \
   "ark:|gzip -c >$alignLatMod/lat.$i.gz" > log.$i.txt
  echo "modify lattice done"
  done
fi 

if [ ! -z $step33 ];then
  for (( i = 1 ; i <= $nj ; i++ ))
  do
##1-best from original lattice
lattice-1best --lm-scale=1 --acoustic-scale=$acwt "ark:gzip -cdf $decodedir/lat.$i.gz|" \
 ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids.$i.txt ark,t:- ark,t:$tmp/lm-costs.$i.txt ark,t:$tmp/acoustic-costs.$i.txt | ../utils/int2sym.pl -f 2- $lang_dir/words.txt > OrgTranscript_${i}_best.hyp

##Following two commands are not very important, I just want to test some thing
lattice-1best --lm-scale=1 --acoustic-scale=$acwt "ark:gzip -cdf $orgLatMod/lat.$i.gz|" \
 ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_mod.$i.txt ark,t:- ark,t:$tmp/lm-costs_mod.$i.txt \
  ark,t:$tmp/acoustic-costs_mod.$i.txt | ../utils/int2sym.pl -f 2- $lang_dir/words.txt > $orgLatMod/ModTranscript_${i}_best.hyp

lattice-1best --lm-scale=1 --acoustic-scale=$acwt "ark:gzip -cdf $alignLat/lat.$i.gz|" \
  ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_align.$i.txt ark,t:- ark,t:$tmp/lm-costs_align.$i.txt \
   ark,t:$tmp/acoustic-costs_align.$i.txt | ../utils/int2sym.pl -f 2- $lang_dir/words.txt > $alignLat/AlignedLattice_${i}_best.hyp

#This important. It estimate the 1-best from the modified lattice
lattice-1best --lm-scale=1 --acoustic-scale=1 "ark:gzip -cdf $alignLatMod/lat.$i.gz|" \
 ark:- | nbest-to-linear ark:- ark,t:$tmp/transition-ids_alignMod.$i.txt ark,t:- ark,t:$tmp/lm-costs_alignMod.$i.txt \
  ark,t:$tmp/acoustic-costs_alignMod.$i.txt | ../utils/int2sym.pl -f 2- $lang_dir/words.txt > $alignLatMod/AlignedLatticeMod_${i}_best.hyp

done
fi


# nnet3 decode,
# I used trained chain model to decode the test dataset and increase beam width from 15(e.g.:default value) to 25
if [ ! -z $step36 ];then
   test_set=test-wavdata-kaldi-format_segmented
   decode_nj=3
   dir=asr_models_english_zhiping/chain1024tdnnf
   graph_dir=asr_models_english_zhiping/chain1024tdnnf/graph-4gram-only-english
   for decode_set in $test_set; do
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 --beam 25 \
          --nj $decode_nj --cmd "${cmd}"  \
          --online-ivector-dir exp/nnet3_1a/ivectors_${decode_set}_hires_zhiping \
          $graph_dir data/${decode_set}_hires \
          $dir/decode_${decode_set}_beam25 || exit 1;
  done
# %WER 44.41 [ 131 / 295, 25 ins, 28 del, 78 sub ] asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_beam25/wer_12_0.0
fi
 
# get oracle lattice from chain model decode test set and oracle wer
# refrence:kaldi/egs/wsj/s5/steps/cleanup/clean_and_segment_data_nnet3.sh
if [ ! -z $step37 ];then
   # some check work
   data=data/test-wavdata-kaldi-format_segmented_hires
   # here $lang function  only offers words.txt
   #lang=asr_models_english_zhiping/lang-4glm-only-english
   lang=asr_models_english_zhiping/lang-only-english
   srcdir=asr_models_english_zhiping/chain1024tdnnf
   dir=asr_models_english_zhiping/chain1024tdnnf/decode_test-wavdata-kaldi-format_segmented_beam25
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

  #utils/lang/check_phones_compatible.sh $lang/phones.txt $srcdir/phones.txt
  #cp $lang/phones.txt $dir

  frame_shift_opt=
  if [ -f $srcdir/frame_subsampling_factor ]; then
    frame_shift_opt="--frame-shift 0.0$(cat $srcdir/frame_subsampling_factor)"
  fi
  echo "$0: Doing oracle alignment of lattices..."
  steps/cleanup/lattice_oracle_align.sh --cmd "$cmd --mem 4G" $frame_shift_opt \
    $data $lang $dir $dir/lattice_oracle
fi
# result:cat log/steps37.log
# overall oracle %WER is: 22.71%


