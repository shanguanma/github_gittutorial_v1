#!/bin/bash

. cmd.sh
. path.sh
set -e

echo
echo "## LOG: $0 $@"
echo
# current work folder:/home4/md510/w2019a/kaldi-recipe/nana_research
# its means the script contains all path is relate the work folder if path isn't absolute Path.


# first, I define this problem,
# 对一段语音进行帧级别的划分，
# 并找到帧对应的phone,然后对phone 进行one-hot 编码，
# 最后把这段语音用一个矩阵表示，
# 矩阵行就这段语音的帧数，矩阵的列就是对应phone 的one-hot 向量,
# one-hot 向量的长度就基本phone 的个数
# 技术上，对one-hot 向量进行扰动，用一个很小的值去代替0

# 对于该问题来说，一段语音就是一句话，
# 基本phone的个数，因为这里做实验的英语音频，所以基本phone 只有位置独立的english Phone（个数是46).
# 也就是说一个one-hot 向量长度是46
# 
# common option
#cmd="slurm.pl --quiet --nodelist=node07"
cmd="slurm.pl  --quiet --exclude=node05"
steps=
nj=25
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

train_set=train_kaldi-format # clean data set .it don't include any noise data.
mkdir -p data/$train_set
# prepare kaldi format from wav folder(e.g:)
# note:clean_trainset_28spk_wav and clean_trainset_56spk is clean data, they don't include any noise
# make wav.scp
data_root_folder_dir=/home4/md510/w2019a/kaldi-recipe/nana_research/original_data
if [ ! -z $step01 ];then
   
   find $data_root_folder_dir/clean_trainset_28spk_wav -name "*.wav" > $data_root_folder_dir/clean_trainset_28spk_wav_list
   find $data_root_folder_dir/clean_trainset_56spk_wav -name "*.wav" > $data_root_folder_dir/clean_trainset_56spk_wav_list
   cat  $data_root_folder_dir/clean_trainset_28spk_wav_list | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > data/$train_set/wav_1.scp
   cat  $data_root_folder_dir/clean_trainset_56spk_wav_list | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > data/$train_set/wav_2.scp
    
   cat data/$train_set/wav_1.scp data/$train_set/wav_2.scp | sort -u >data/$train_set/wav.scp
   
fi


train_dir=data/$train_set
if [ ! -z $step02 ];then
   #  make utt2spk spk2utt
   cat $train_dir/wav.scp | awk '{print $1, $1;}' > $train_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $train_dir/utt2spk > $train_dir/spk2utt
fi
if [ ! -z $step03 ];then
   # make text
   find $data_root_folder_dir/trainset_28spk_txt -name "*.txt"> $data_root_folder_dir/trainset_28spk_txt_list
   find $data_root_folder_dir/trainset_56spk_txt -name "*.txt"> $data_root_folder_dir/trainset_56spk_txt_list
   cat $data_root_folder_dir/trainset_28spk_txt_list $data_root_folder_dir/trainset_56spk_txt_list > $data_root_folder_dir/trainset_84spk_txt_list 
   cat $data_root_folder_dir/trainset_84spk_txt_list \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data_root_folder_dir/text_scp_trainset.txt
   
   # corvet text_scp to kaldi text
   source-md/egs/nana_research/make_text.py $data_root_folder_dir/text_scp_trainset.txt $data_root_folder_dir/pre_text
   # remove punctuation in $data_root_folder_dir/pre_text
   
   sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data_root_folder_dir/pre_text | sed 's/[A-Z]/\l&/g' | sort >$train_dir/text
fi
# At this point, I have finished dealing with the clean voice kaldi format.
# Below I started to deal with the kaldi format of clean data with noise data.

# first make wav.scp
# current folder path:/home4/md510/w2019a/kaldi-recipe/nana_research
# note:nosisy_set is Nosiy data, it  is the data after the clean noise adds the corresponding noise.
# What noise is added to each audio, how much is the signal to noise ratio, which is explained in detail in these document.
# (e.g:original_data/logfiles/log_readme.txt  log_testset.txt  log_trainset_28spk.txt  log_trainset_56spk.txt).
noisy_set=nosiy_kaldi_format # have added noisy data set
mkdir -p data/$noisy_set
if [ ! -z $step04 ];then
  find $data_root_folder_dir/noisy_trainset_28spk_wav -name "*.wav" >$data_root_folder_dir/noisy_trainset_28spk_wav_list
  find $data_root_folder_dir/noisy_trainset_56spk_wav -name "*.wav" >$data_root_folder_dir/noisy_trainset_56spk_wav_list
  cat  $data_root_folder_dir/noisy_trainset_28spk_wav_list | \
  perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > data/$noisy_set/wav_1.scp
  cat  $data_root_folder_dir/noisy_trainset_56spk_wav_list | \
  perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > data/$noisy_set/wav_2.scp
  #cat data/$noisy_set/wav_1.scp data/$noisy_set/wav_2.scp | sort -u >data/$noisy_set/wav.scp
  #rm -rf  data/$noisy_set/wav_1.scp 
  #rm -rf data/$noisy_set/wav_2.scp
 
fi

noisy_dir=data/$noisy_set
if [ ! -z $step05 ];then
   #  make utt2spk spk2utt
   cat $noisy_dir/wav.scp | awk '{print $1, $1;}' > $noisy_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $noisy_dir/utt2spk > $noisy_dir/spk2utt
fi

if [ ! -z $step06 ];then
   # make text 
   # Their transcription is the same whether you add noise or clean audio.
   # remove punctuation in $data_root_folder_dir/pre_text and covert upper to lower
   sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data_root_folder_dir/pre_text | sed 's/[A-Z]/\l&/g' | sort > $noisy_dir/text

fi
# At this point, I have finished dealing with the clean voice with adding noise data.Then getting kaldi format.
# kaldi format includes wav.scp text utt2spk spk2utt

# Next ,I used $noisy _set as train set to train gmm-hmm system
# Unless otherwise specified, train set(e.g:$noisy _set) refers to a data set that has been noisy.
# The overall process is as follows:
# I start to prepare lang lang_test and hmm-gmm system.
# first, I start to prepare lang
# second , I  use pure english dict to check oov of the train set.
# note: pure english dict is from /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/dict-only-english. 

# make lang
if [ ! -z $step07 ];then
  dict=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/dict-only-english
  lang=data/lang
  utils/prepare_lang.sh $dict "<unk>" $lang/tmp $lang || exit 1;
 
fi

# chech oov from train set(e.g:$noisy _set).

if [ ! -z $step08 ];then

  # First find a list of words in the test set that are out of vocabulary.
  # Of course this is totally cheating.
  oov_file=data/oov.txt
  awk -v w=data/lang/words.txt 'BEGIN{while(getline <w) seen[$1] = $1} {for(n=2;n<=NF;n++) if(!($n in seen)) oov[$n] = 1}
                                END{ for(k in oov) print k;}' < data/nosiy_kaldi_format/text > $oov_file
  echo "$0: generating g2p entries for $(wc -l <$oov_file) words"
  # remove the symbol (e.g: -)
  cat $oov_file | sed 's/^-//g'| sed '/^$/d'>data/oov_1.txt
fi
# make g2p model
if [ ! -z $step09 ];then
  dict=/home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/dict-only-english
  g2p_dir=data/g2p
  steps/dict/train_g2p.sh --silence-phones $dict/silence_phones.txt $dict/lexicon.txt  $g2p_dir

fi

# getting a dictionary with oov words
if [ ! -z $step10 ];then
  oov_file=data/oov_1.txt  # oov words list
  g2p_dir=data/g2p       # g2p model 
  g2p_dict=data/g2p_dict # apply the g2p model to get the new dictionary.
  steps/dict/apply_g2p.sh $oov_file $g2p_dir  $g2p_dict
fi

# make new dictionary, it includes oov words.
if [ ! -z $step11 ];then

  new_dict=data/dict_add_oov
  mkdir -p $new_dict
  cp -r /home4/md510/w2019a/kaldi-recipe/chng_project_2019/asr_models_english_zhiping/dict-only-english/* $new_dict
  #cp -r data/g2p_dict/lexicon.lex $new_dict/
  #cat $new_dict/lexiconp.txt data/g2p_dict/lexicon.lex  | sort | uniq > $new_dict/lexiconp_new.txt
  rm -rf  $new_dict/lexiconp.txt
  #mv  $new_dict/lexiconp_new.txt  $new_dict/lexiconp.txt
  awk 'BEGIN{FS="\t"} {print $1 , $3}' data/g2p_dict/lexicon.lex > $new_dict/lexicon_oov.txt 
  
  cat $new_dict/lexicon_oov.txt $new_dict/lexicon.txt | sort | uniq>$new_dict/lexicon_new.txt 
  rm -rf $new_dict/lexicon.txt
  rm -rf  $new_dict/lexicon_oov.txt
  mv $new_dict/lexicon_new.txt $new_dict/lexicon.txt
   utils/validate_dict_dir.pl $new_dict
fi

# make new lang
new_dict=data/dict_add_oov
new_lang=data/lang_new
if [ ! -z $step12 ];then
  utils/prepare_lang.sh $new_dict "<unk>" $new_lang/tmp $new_lang || exit 1;
 
fi

###############################################################################
#Now make MFCC features in add augment seame supervised data (seame).
###############################################################################
# because data/nosiy_kaldi_format is 48k audio ,so I need to  downsample it to 16k.
# make downsample wav.scp
if [ ! -z $step13 ];then
sox=`which sox`
if [ $? -ne 0 ] ; then
  echo "Could not find sox binary. Add it to PATH"  
  exit 1;
fi
datadir=data/nosiy_kaldi_format
audiopath=/home4/md510/w2019a/kaldi-recipe/nana_research/original_data
echo "Creating the $datadir/wav_1_downsample.scp file"
(
  set -o pipefail
  for file in `cut -f 1 -d ' ' $datadir/wav_1.scp` ; do
    echo "$file $sox $audiopath/noisy_trainset_28spk_wav/$file.wav -r 16000 -c 1 -b 16 -t wav - downsample |"
 
  done | sort -u > $datadir/wav_1_downsample.scp 
  if [ $? -ne 0 ] ; then 
    echo "Error producing the wav_1_downsample.scp file" 
    exit 1
  fi
) || exit 1

echo "Creating the $datadir/wav_2_downsample.scp file"
(
  set -o pipefail
  for file in `cut -f 1 -d ' ' $datadir/wav_2.scp` ; do
    echo "$file $sox $audiopath/noisy_trainset_56spk_wav/$file.wav -r 16000 -c 1 -b 16 -t wav - downsample |" 
  
  done | sort -u > $datadir/wav_2_downsample.scp 
  if [ $? -ne 0 ] ; then 
    echo "Error producing the wav.scp file" 
    exit 1
  fi
) || exit 1

cat $datadir/wav_1_downsample.scp  $datadir/wav_2_downsample.scp |  sort -u >$datadir/wav.scp 
fi

mfccdir=`pwd`/mfcc
if [ ! -z $step14 ];then
  for sdata in nosiy_kaldi_format; do
    # this must be make_mfcc ,it shouldn't  add pitch. otherwise running steps/align_fmllr_lats.sh in step18 local/semisup/chain/run_tdnn.sh is error.  
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
      --mfcc-config conf/mfcc.conf  --write-utt2num-frames true data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    #This script will fix sorting errors and will remove any utterances for which some required data, such as feature data or transcripts, is missing.
    utils/fix_data_dir.sh data/${sdata}
    echo "## LOG : done with 13 dimension mfcc feat"
    done
fi

###############################################################################
# Prepare the add augment seame supervised set and subsets for initial GMM training
###############################################################################

if [ ! -z $step15 ]; then
  utils/subset_data_dir.sh  data/nosiy_kaldi_format 10000 data/nosiy_kaldi_format_10k

fi
# train monophone
if [ ! -z $step16 ]; then
  steps/train_mono.sh --nj  $nj  --cmd "$cmd" \
    data/nosiy_kaldi_format_10k $new_lang $exp_root/mono0a || exit 1
fi
# ali monophone
# using 2500 number leaves and 20000 gaussion to  train triphone
if [ ! -z $step17 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    data/nosiy_kaldi_format $new_lang $exp_root/mono0a $exp_root/mono0a_ali || exit 1

  steps/train_deltas.sh --cmd "$cmd" \
    2500 20000 data/nosiy_kaldi_format $new_lang $exp_root/mono0a_ali $exp_root/tri1 || exit 1
fi
# ali triphone
# using 2500 number leaves and 20000 gaussion to train lda_mllt triphone
if [ ! -z $step18 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
   data/nosiy_kaldi_format $new_lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
    2500 20000 data/nosiy_kaldi_format $new_lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

fi
# ali triphone
# using 5000 number leaves and 40000 gaussion to train lda_mllt triphone

if [ ! -z $step19 ]; then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    data/nosiy_kaldi_format $new_lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;

  steps/train_lda_mllt.sh --cmd "$cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 data/nosiy_kaldi_format $new_lang $exp_root/tri2_ali $exp_root/tri3a || exit 1;

fi
# ali triphone
# do Speaker Adapted Training (SAT) for triphone system
if [ ! -z $step20 ]; then
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    data/nosiy_kaldi_format $new_lang $exp_root/tri3a $exp_root/tri3a_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 data/nosiy_kaldi_format $new_lang $exp_root/tri3a_ali $exp_root/tri4a || exit 1;

fi

# get ctm from ali model (e.g: tri4a)
# here the script gets word boundary info in $exp_root/tri4a/ctm
if [ ! -z $step21 ];then
   new_lang=data/lang_new # it is a lang
   steps/get_train_ctm.sh data/nosiy_kaldi_format $new_lang $exp_root/tri4a 
fi
# get ctm_phone from ali model (e.g: tri4a)
if [ ! -z $step22 ];then
  # refrence:egs/wsj/s5/steps/segmentation/ali_to_targets.sh
  ali_dir=$exp_root/tri4a
  dir=$ali_dir
  nj=$(cat $ali_dir/num_jobs) || exit 1
  new_lang=data/lang_new # it is a lang
  
  $cmd JOB=1:$nj $dir/log/get_ctm_phone.JOB.log \
  ali-to-phones --ctm-output \
    $ali_dir/final.mdl "ark:gunzip -c $ali_dir/ali.JOB.gz |" - \| \
  utils/int2sym.pl -f 5 $new_lang/phones.txt \| \
  gzip -c '>' $dir/ctm_phone.JOB.gz || exit 1 
   
  for n in `seq $nj`; do gunzip -c $dir/ctm_phone.$n.gz; done > $dir/ctm_phone || exit 1; 
  #awk '{print $1" "int($3)" "int($4)" 1.0 "$5}' \> \
  #$dir/arc_info_sym.JOB.txt || exit 1
  rm $dir/ctm_phone.*.gz
 
fi

# get phone posterior
# refrence:steps/nnet3/chain/get_phone_post.sh 
dir=exp/tri4a_phone
mkdir -p $dir
ali_dir=exp/tri4a
# get transition-id count
if [ ! -z $step23 ];then
  if [ ! -f $dir/tacc ] || [ $dir/tacc -ot $ali_dir/ali.1.gz ]; then
    echo "$0: obtaining transition-id counts in $dir/tacc"
    # Obtain counts for each transition-id, from the alignments.
    this_nj=$(cat $ali_dir/num_jobs)

    $cmd JOB=1:$this_nj $dir/log/acc_taccs.JOB.log \
       ali-to-post "ark:gunzip -c $ali_dir/ali.JOB.gz|" ark:- \| \
       post-to-tacc $ali_dir/final.mdl ark:- $dir/tacc.JOB

    input_taccs=$(for n in $(seq $this_nj); do echo $dir/tacc.$n; done)

    $cmd $dir/log/sum_taccs.log \
         vector-sum --binary=false $input_taccs $dir/tacc

    rm $dir/tacc.*
  else
    echo "$0: skipping creation of $dir/tacc since it already exists."
  fi
fi

# Get a map. This map is a position-dependent phone to a positon-independent phone map
if [ ! -z $step24 ];then
  lang=data/lang_new
  remove_word_position_dependency=true
  if $remove_word_position_dependency; then
  echo "$0: creating $dir/phone_map.int"
  utils/lang/get_word_position_phone_map.pl $lang $dir
  else
  # Either way, $dir/phones.txt will be a symbol table for the phones that
  # we are dumping (although the matrices we dump won't contain anything
  # for symbol 0 which is <eps>).
  grep -v '^#' $lang/phones.txt > $dir/phones.txt
  fi
fi

# # note: steps25-27,It has nothing to do with solving the problem, I am just doing some testing. 
# get a map .This map is a postion-dependent phone to a pdf map
if [ ! -z $step25 ]; then
  lang=data/lang_new
  # we want the phones in integer form as it's safer for processing by script.
  # $data/fake_phones.txt will just contain e.g. "0 0\n1 1\n....", it's used
  # to force show-transitions to print the phones as integers.
  awk '{print $2,$2}' <$lang/phones.txt >$dir/fake_phones.txt

  # chain model case:
  # The format of the 'show-transitions' command below is like the following:
  #show-transitions tempdir/phone_map.int exp/chain/tree_a_sp/final.mdl
  #Transition-state 1: phone = 1 hmm-state = 0 forward-pdf = 0 self-loop-pdf = 51
  # Transition-id = 1 p = 0.5 [self-loop]
  # Transition-id = 2 p = 0.5 [0 -> 1]
  #Transition-state 2: phone = 10 hmm-state = 0 forward-pdf = 0 self-loop-pdf = 51
  # Transition-id = 3 p = 0.5 [self-loop]
  # Transition-id = 4 p = 0.5 [0 -> 1]

  # The following inline script processes that info about the transition model
  # into the file $dir/phones_and_pdfs.txt, which has a line for each transition-id
  # (starting from number 1), and the format of each line is
  # <phone-id> <pdf-id>
  # the following command:
  #show-transitions $dir/fake_phones.txt $ali_dir/final.mdl | \
  #  perl -ane ' if(m/Transition-state.* phone = (\d+) pdf = (\d+)/) { $phone = $1; $forward_pdf = $2; $self_loop_pdf = $2; }
  #      if(m/Transition-state.* phone = (\d+) .* forward-pdf = (\d+) self-loop-pdf = (\d+)/) {
  #        $phone = $1; $forward_pdf = $2; $self_loop_pdf = $3; }
  #      if(m/Transition-id/) {  if (m/self-loop/) { print "$phone $self_loop_pdf\n"; }
  #          else { print "$phone $forward_pdf\n" } } ' > $dir/phones_and_pdfs.txt

  # gmm-hmm model case:
  #The format of the 'show-transitions' command below is like the following:
  #show-transitions $dir/fake_phones.txt exp/tri4a/final.mdl
  
  # Transition-state 16883: phone = 1028 hmm-state = 0 pdf = 255
  # Transition-id = 33925 p = 0.75 [self-loop]
  # Transition-id = 33926 p = 0.25 [0 -> 1]
  # Transition-state 16884: phone = 1028 hmm-state = 1 pdf = 255
  # Transition-id = 33927 p = 0.75 [self-loop]
  # Transition-id = 33928 p = 0.25 [1 -> 2]
  # Transition-state 16885: phone = 1028 hmm-state = 2 pdf = 255
  # Transition-id = 33929 p = 0.75 [self-loop]
  # Transition-id = 33930 p = 0.25 [2 -> 3]
  
  # The following inline script function is covert transition model to $dir/phones_and_pdfs.txt
  # phones_and_pdfs.txt format is as follows:
  # <phone-id> <pdf-id>
  show-transitions $dir/fake_phones.txt $ali_dir/final.mdl | \
    perl -ane ' if(m/Transition-state.* phone = (\d+) pdf = (\d+)/) { $phone = $1; $pdf = $2; }
        if(m/Transition-state.* phone = (\d+) .* pdf = (\d+)/) {
          $phone = $1; $pdf = $2; }
        if(m/Transition-id/) {  print "$phone $pdf\n";} '|sort -k 1n > $dir/phones_and_pdfs.txt

fi 
# get a phone one-hot encode in $dir/transform.mat
count_smoothing=1.0  # this should be some small number, I don't think it's critical;
                     # it will mainly affect the probability we assign to phones that
                     # were never seen in training.  note: this is added to the raw
                     # transition-id occupation counts, so 1.0 means, add a single
                     # frame's count to each transition-id's counts.

if [ ! -z $step26 ];then
  # The following command just separates the 'tacc' file into a similar format
  # to $dir/phones_and_pdfs.txt, with one count per line, and a line per transition-id
  # starting from number 1.  We skip the first two fields which are "[ 0" (the 0 is
  # for transition-id=0, since transition-ids are 1-based), and the last field which is "]".
  awk '{ for (n=3;n<NF;n++) print $n; }' <$dir/tacc  >$dir/transition_counts.txt

  num_lines1=$(wc -l <$dir/phones_and_pdfs.txt)
  num_lines2=$(wc -l <$dir/transition_counts.txt)
  if [ $num_lines1 -ne $num_lines2 ]; then
    echo "$0: mismatch in num-lines between phones_and_pdfs.txt and transition_counts.txt: $num_lines1 vs $num_lines2"
    exit 1
  fi

  # after 'paste', the format of the data will be
  # <phone-id> <pdf-id> <data-count>
  # we add the count smoothing at this point.
  paste $dir/phones_and_pdfs.txt $dir/transition_counts.txt | \
     awk -v s=$count_smoothing '{print $1, $2, (s+$3);}' > $dir/combined_info.txt

  if $remove_word_position_dependency; then
    # map the phones to word-position-independent phones; you can see $dir/phones.txt
    # to interpret the final output.
    utils/apply_map.pl -f 1 $dir/phone_map.int <$dir/combined_info.txt > $dir/temp.txt
    mv $dir/temp.txt $dir/combined_info.txt
  fi

  awk 'BEGIN{num_phones=1;num_pdfs=1;} { phone=$1; pdf=$2; count=$3; pdf_count[pdf] += count; counts[pdf,phone] += count;
       if (phone>num_phones) num_phones=phone; if (pdf>=num_pdfs) num_pdfs = pdf + 1; }
       END{ print "[ "; for(phone=1;phone<=num_phones;phone++) {
          for (pdf=0;pdf<num_pdfs;pdf++) printf("%.3f ", counts[pdf,phone]/pdf_count[pdf]);
           print ""; } print "]"; }' <$dir/combined_info.txt >$dir/transform.mat

fi


# get phone sequence in a utterance from ali model (e.g: tri4a)
if [ ! -z $step27 ];then
  # refrence:egs/wsj/s5/steps/segmentation/ali_to_targets.sh
  ali_dir=$exp_root/tri4a
  dir=$ali_dir
  nj=$(cat $ali_dir/num_jobs) || exit 1
  new_lang=data/lang_new # it is a lang

  $cmd JOB=1:$nj $dir/log/get_phone_sequence.JOB.log \
  ali-to-phones \
    $ali_dir/final.mdl "ark:gunzip -c $ali_dir/ali.JOB.gz |" ark,t:- \|\
  utils/int2sym.pl -f 2- $new_lang/phones.txt \| \
  gzip -c '>' $dir/phone_sequence.JOB.gz || exit 1

  for n in `seq $nj`; do gunzip -c $dir/phone_sequence.$n.gz; done > $dir/phone_sequence || exit 1;
  #awk '{print $1" "int($3)" "int($4)" 1.0 "$5}' \> \
  #$dir/arc_info_sym.JOB.txt || exit 1
  rm $dir/phone_sequence.*.gz

fi

# note: steps25-27,It has nothing to do with solving the problem, I am just doing some testing. 
# Get a map. This map is a position-dependent phone to a positon-independent phone map
if [ ! -z $step28 ];then
  lang=data/lang_new
  sub_dir=exp/tri4a_phone_sub
  mkdir -p $sub_dir
  # get base english phone set ,and numbered from zero
  grep -vP '_man|#|<eps>' exp/tri4a_phone/phones.txt | awk '{print $1,NR-1}' > $sub_dir/position_independent_phone_set.txt
  # map position dependent english phone to position independent phone set
  grep -vP '_man|#|<eps>' exp/tri4a_phone/phone_map.txt  > $sub_dir/english_phone_map.txt
  
fi
# get a map about postion_dependent phone mark
if [ ! -z $step29 ];then
  sub_dir=exp/tri4a_phone_sub
  source-md/egs/nana_research/make_id_for_position_phone.py $sub_dir/position_independent_phone_set.txt $sub_dir/english_phone_map.txt $sub_dir/postion_dependent_phone_mark.txt

fi
# get phone2frames
if [ ! -z $step30 ];then  
  awk '{print $1,$2,($3/0.01),($4/0.01),$5}' exp/tri4a/ctm_phone > $sub_dir/ctm_phone2frames

fi

if [ ! -z $step31 ];then
  sub_dir=exp/tri4a_phone_sub
  source-md/egs/nana_research/ctm_position_phone_mark.py $sub_dir/postion_dependent_phone_mark.txt $sub_dir/ctm_phone2frames $sub_dir/ctm_phone2frames_mark
fi

# I prepared two kind of phone feat array
# case 1:
# the details of phone feat :
# .scp file format is as follows:
# <recod-id:position_independent_phone_id> <path of the phone array>
# .ark file format is as follows:
# <recod-id:position_independent_phone_id> <content of the phone array >
# note:<content of the phone array> : array row is frames of the phone.
#                                     array column is a one-hot style vector, vector length is the number of position_independent_phones set 
#                                      (e.g:it (/home4/md510/w2019a/kaldi-recipe/nana_research/exp/tri4a_phone_sub/position_independent_phone_set.txt)
#                                      is a pure english position independent phone set. the set contain 46 phones,so the vetor length is 46) 
# what is one-hot style vector?
# it is based on the one-hot vector, but I replaced zeros with a very small position random numbers.
# you can see code:source-md/egs/nana_research/make_phone_feat_test.py
# raw .scp file and .ark file are stored in/home4/md510/w2019a/kaldi-recipe/nana_research/exp/make_phone_feats_for_tri4a_phone_sub

# the follows is demo test.
# make $phone_featsdir an absolute pathname.
phone_featsdir=exp/make_phone_feats_for_tri4a_phone_sub # raw feats folder
phone_featsdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${phone_featsdir} ${PWD})


logdir=exp/phone_feats_log
mkdir -p ${phone_featsdir} || exit 1;
mkdir -p ${logdir} || exit 1;

#split ctm_phone2frames_mark_48,  per 12 line as a file
if [ ! -z $step32 ];then
   lines=12
   awk -v dir="$logdir" -v lines=$lines 'NR%lines==1{x=dir"/ctm_phone2frames_mark_48_"++i;}{print >x}' ctm_phone2frames_mark_48
fi

if [ ! -z $step33 ];then
   nj=4
   ${cmd} JOB=1:${nj} ${logdir}/make_phone_feats_demo.JOB.log \
   python3 source-md/egs/nana_research/make_phone_feat_test.py \
     ${logdir}/ctm_phone2frames_mark_48_JOB \
     ark,t,scp:${phone_featsdir}/raw_phone_feats_demo.JOB.ark,${phone_featsdir}/raw_phone_feats_demo.JOB.scp
fi
if [ ! -z $step34 ];then
   # concatenate the .scp files together.
   feats_dir=exp/tri4a_phone_sub
   for n in $(seq ${nj}); do
      cat ${phone_featsdir}/raw_phone_feats_demo.${n}.scp || exit 1;
   done > ${feats_dir}/feats_demo.scp || exit 1
fi

# split file
if [ ! -z $step35 ];then
   # cat exp/tri4a_phone_sub/ctm_phone2frames_mark | wc -l
   # 975250
   # 975250÷25=39010 
   lines=39010 # 
   awk -v dir="$logdir" -v lines=$lines 'NR%lines==1{x=dir"/ctm_phone2frames_mark_"++i;}{print >x}' exp/tri4a_phone_sub/ctm_phone2frames_mark

fi

if [ ! -z $step36 ];then
   nj=25
   ${cmd} JOB=1:${nj} ${logdir}/make_phone_feats.JOB.log \
   python3 source-md/egs/nana_research/make_phone_feat_test.py \
     ${logdir}/ctm_phone2frames_mark_JOB \
     ark,t,scp:${phone_featsdir}/raw_phone_feats.JOB.ark,${phone_featsdir}/raw_phone_feats.JOB.scp
fi
if [ ! -z $step37 ];then
   # concatenate the .scp files together.
   feats_dir=exp/tri4a_phone_sub
   for n in $(seq ${nj}); do
      cat ${phone_featsdir}/raw_phone_feats.${n}.scp || exit 1;
   done > ${feats_dir}/feats.scp || exit 1
fi

# case 2
# the details of phone feat :
# .scp file format is as follows:
# <recod-id> <path of the utterance array>
# .ark file format is as follows:
# <recod-id> <content of the utterance array >
# note:<content of the utterance array> : array row is frames of the utterance.Keep the position of the phone before and after the phone.
#                                        In other words, Here the frames are arranged in order of time.
#                                     array column is a one-hot style vector, vector length is the number of position_independent_phones set 
#                                      (e.g:it (/home4/md510/w2019a/kaldi-recipe/nana_research/exp/tri4a_phone_sub/position_independent_phone_set.txt)
#                                      is a pure english position independent phone set. the set contain 46 phones,so the vetor length is 46) 
# what is one-hot style vector?
# it is based on the one-hot vector, but I replaced zeros with a very small position random numbers.
# you can see code:source-md/egs/nana_research/make_phone_feat_test.py
# raw .scp file and .ark file are stored in/home4/md510/w2019a/kaldi-recipe/nana_research/exp/make_phone_feats_for_tri4a_phone_sub
phone_featsdir=exp/make_phone_feats_for_tri4a_phone_sub_v1 # raw feats folder
phone_featsdir=$(perl -e '($dir,$pwd)= @ARGV; if($dir!~m:^/:) { $dir = "$pwd/$dir"; } print $dir; ' ${phone_featsdir} ${PWD})


logdir=exp/phone_feats_log_v1
mkdir -p ${phone_featsdir} || exit 1;
mkdir -p ${logdir} || exit 1;

#split ctm_phone2frames_mark_48,  per 12 line as a file
if [ ! -z $step38 ];then
   lines=12
   awk -v dir="$logdir" -v lines=$lines 'NR%lines==1{x=dir"/ctm_phone2frames_mark_48_"++i;}{print >x}' ctm_phone2frames_mark_48
fi

if [ ! -z $step39 ];then
   nj=4
   ${cmd} JOB=1:${nj} ${logdir}/make_phone_feats_demo.JOB.log \
   python3 source-md/egs/nana_research/make_phone_feat_test_v1.py \
     ${logdir}/ctm_phone2frames_mark_48_JOB \
     ark,t,scp:${phone_featsdir}/raw_phone_feats_demo.JOB.ark,${phone_featsdir}/raw_phone_feats_demo.JOB.scp
fi
if [ ! -z $step40 ];then
   # concatenate the .scp files together.
   feats_dir=exp/tri4a_phone_sub
   for n in $(seq ${nj}); do
      cat ${phone_featsdir}/raw_phone_feats_demo.${n}.scp || exit 1;
   done > ${feats_dir}/feats_demo_v1.scp || exit 1
fi

# split file
if [ ! -z $step41 ];then
   # cat exp/tri4a_phone_sub/ctm_phone2frames_mark | wc -l
   # 975250
   # 975250÷25=39010 
   lines=39010 # 
   awk -v dir="$logdir" -v lines=$lines 'NR%lines==1{x=dir"/ctm_phone2frames_mark_"++i;}{print >x}' exp/tri4a_phone_sub/ctm_phone2frames_mark

fi
if [ ! -z $step42 ];then
   nj=25
   ${cmd} JOB=1:${nj} ${logdir}/make_phone_feats.JOB.log \
   python3 source-md/egs/nana_research/make_phone_feat_test_v1.py \
     ${logdir}/ctm_phone2frames_mark_JOB \
     ark,t,scp:${phone_featsdir}/raw_phone_feats.JOB.ark,${phone_featsdir}/raw_phone_feats.JOB.scp
fi
if [ ! -z $step43 ];then
   # concatenate the .scp files together.
   feats_dir=exp/tri4a_phone_sub
   for n in $(seq ${nj}); do
      cat ${phone_featsdir}/raw_phone_feats.${n}.scp || exit 1;
   done > ${feats_dir}/feats_v1.scp || exit 1
fi


# 2020-4-7
# in order to test ASR performance, I will decode dev_set by use tri4 model.
# make kaldi format for test set (e.g: dev_test)
dev_test_dir=data/dev_test
dev_test_audio_root=/data/users/nana511/vctk/vctk_vad/corpus_1s/wav_fake16_novad_1s_tt
dev_test_text_root=/data/users/nana511/vctk/txt
if [ ! -z $step44 ];then
   # make wav.scp
   for data_set in p347  p351  p360  p361  p362  p363  p364  p374  p376;do 
      find $dev_test_audio_root/$data_set -name "*.wav" >> $dev_test_dir/wav_list.txt  
   done
   cat  $dev_test_dir/wav_list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  | sort > $dev_test_dir/wav.scp
   #  make utt2spk spk2utt
   cat  $dev_test_dir/wav.scp | awk '{print $1, $1;}' >  $dev_test_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl  $dev_test_dir/utt2spk >  $dev_test_dir/spk2utt
fi
# make text file for test set(e.g: dev_test)
if [ ! -z $step45 ];then
   for data_set in p347  p351  p360  p361  p362  p363  p364  p374  p376; do
     find $dev_test_text_root/$data_set -name "*.txt">> $dev_test_dir/txt_list.txt
   done
   cat $dev_test_dir/txt_list.txt | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  >  $dev_test_dir/text_scp.txt

   # corvet text_scp to kaldi text
   source-md/egs/nana_research/make_text.py  $dev_test_dir/text_scp.txt  $dev_test_dir/pre_text
   # remove punctuation in $data_root_folder_dir/pre_text

   sed --regexp-extended 's/[,|\.|?|)|"|!]//g'  $dev_test_dir/pre_text | sed 's/[A-Z]/\l&/g' | sort > $dev_test_dir/text 
fi
# make 13 dim mfccc for dev_test
if [ ! -z $step46 ];then
  for sdata in dev_test; do
    # this must be make_mfcc ,it shouldn't  add pitch. otherwise running steps/align_fmllr_lats.sh in step18 local/semisup/chain/run_tdnn.sh is error.  
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
      --mfcc-config conf/mfcc.conf  --write-utt2num-frames true data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    #This script will fix sorting errors and will remove any utterances for which some required data, such as feature data or transcripts, is missing.
    utils/fix_data_dir.sh data/${sdata}
    echo "## LOG : done with 13 dimension mfcc feat"
    done

fi 

###############################################################################
# pepared new lang_test 
# more detail ,you can read /home4/md510/w2019a/kaldi-recipe/lm/data/lm/perplexities.txt
#                           /home4/md510/w2019a/kaldi-recipe/lm/data/lm/ 
#                           /home4/md510/w2019a/kaldi-recipe/lm/train_lms_srilm.sh 
###############################################################################
datadir=data
lmdir=data/local/lm
train_data=${datadir}/nosiy_kaldi_format 
# prepared G.fst
[ -d $lmdir ] || mkdir -p $lmdir
oov_symbol="<UNK>"
new_dict=data/dict_add_oov
words_file=data/lang_new/words.txt
if [ ! -z $step47 ]; then
   echo "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat $train_data/text | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ! -z $step48 ]; then
  echo "-------------------"
  echo "Maxent 4grams"
  echo "-------------------"
  # sed 's/'${oov_symbol}'/<unk>/g' means: using <unk> to replace ${oov_symbol}
  sed 's/'${oov_symbol}'/<unk>/g' $lmdir/train.txt | \
    ngram-count -lm - -order 4 -text - -vocab $lmdir/vocab -unk -sort -maxent -maxent-convert-to-arpa| \
   sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > $lmdir/4gram.me.gz || exit 1
  echo "## LOG : done with '$lmdir/4gram.me.gz'"
fi


lang_test=data/local/lang_test
[ -d $lang_test ] || mkdir -p $lang_test

if [ ! -z $step49 ]; then
  utils/format_lm.sh data/lang_new data/local/lm/4gram.me.gz \
    $new_dict/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  echo "## LOG : done with '$lang_test'"
fi



# make HCLG.fst graph
if [ ! -z $step50 ];then
   utils/mkgraph.sh $lang_test \
      $exp_root/tri4a  $exp_root/tri4a/graph || exit 1;
fi
# decode for dev_test
if [ ! -z $step51 ];then
   #nspk=$(wc -l <data/dev_test/spk2utt)
   steps/decode_fmllr.sh --nj 50 --cmd "$cmd" \
      $exp_root/tri4a/graph data/dev_test \
      $exp_root/tri4a/decode_dev_test || exit 1; 

fi
