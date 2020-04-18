#!/bin/bash

. path_v2.sh
. cmd.sh

# this recipe path is /home4/md510/w2019a/espnet-recipe/haoxiang_research 
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

# general configuration
backend=pytorch
steps=1
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
lm_ngpu=1 # LM training does not support multi-gpu. signle gpu will be used.
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot

# feature configuration
do_delta=false

# rnnlm related
lm_resume= # specify a snapshot file to resume LM training
lmtag=     # tag for managing LMs

# decoding parameter
recog_model=model.acc.best  # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'
lang_model=rnnlm.model.best # set a language model to be used for decoding

# model average realted (only for transformer)
n_average=5                  # the number of ASR models to be averaged
use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
lm_n_average=6               # the number of languge models to be averaged
use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# bpemode (unigram or bpe)
nbpe=5000
bpemode=unigram
# exp tag
tag="" # tag for managing experiments.
. ./utils/parse_options.sh || exit 1;

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

#case 1:
# /home/haoxiang215/workspace/audio/data/reformat_data/src5
#case 2:
# /home/haoxiang215/workspace/audio/data/reformat_data/2srcf
#case 3:
# /home/haoxiang215/workspace/audio/data/reformat_data/per_gan

#other people:
# /home/haoxiang215/workspace/audio/data/reformat_data/src0
# make wav.scp
if [ ! -z $step01 ];then
 data_1=data/src5
 srcdata_1=/home/haoxiang215/workspace/audio/data/reformat_data/src5
 [ -d $data_1 ] || mkdir -p $data_1
      find $srcdata_1 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " " > $data_1/wav.scp

 
 data_2=data/2srcf
 srcdata_2=/home/haoxiang215/workspace/audio/data/reformat_data/2srcf
 [ -d $data_2 ] || mkdir -p $data_2

      find $srcdata_2 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_2/wav.scp
  

 data_3=data/per_gan
 srcdata_3=/home/haoxiang215/workspace/audio/data/reformat_data/per_gan
 [ -d $data_3 ] || mkdir -p $data_3
      find $srcdata_3 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_3/wav.scp
  
 # other people's extendtion bank method
 other=data/src0
 srcother=/home/haoxiang215/workspace/audio/data/reformat_data/src0
 [ -d $other ] || mkdir -p $other
      find $srcother -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $other/wav.scp
  

fi

data_1=data/src5
data_2=data/2srcf
data_3=data/per_gan
other=data/src0
if [ ! -z $step02 ];then
   #  make utt2spk spk2utt
   for data in $data_1 $data_2 $data_3 $other;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step03 ];then
   # make text
   for data in $data_1 $data_2 $data_3 $other;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi

# make input data format for recognition
recog_set="src5 2srcf per_gan src0"
# make features 
if [ ! -z $step04 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
if [ ! -z $step05 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi

# 
if [ ! -z $step06 ];then
  # copy from train_960 from /home3/md510/w2019a/kaldi-recipe/librispeech/s5/data/train_960 
  [ -d data/${train_set}_org ] || mkdir -p data/${train_set}_org 
  cp -r /home3/md510/w2019a/kaldi-recipe/librispeech/s5/data/train_960/* data/${train_set}_org/
  # remove utt having more than 3000 frames
  # remove utt having more than 400 characters
  remove_longshortdata.sh --maxframes 3000 --maxchars 400 data/${train_set}_org data/${train_set}_1 
  cp -r data/${train_set}_1/text data/${train_set}
fi


# make bpemodel
dict=data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
if [ ! -z $step07 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p data/lang_char/
    
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    cut -f 2- -d" " data/${train_set}/text > data/lang_char/input.txt
    spm_train --input=data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece < data/lang_char/input.txt | tr ' ' '\n' | sort | uniq | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

fi
# make josn format 
if [ ! -z $step08 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi


if [ ! -z $step09 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml 
    #lm_config=conf/lm.yaml
#    expname=${train_set}_${backend}_$(basename ${train_config%.*})
#    expdir=exp/${expname}
#    [ -d $expdir ] || mkdir -p ${expdir}
#    ###add character lm
#    if [ -z ${lmtag} ]; then
#    lmtag=$(basename ${lm_config%.*})
#    fi
#    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
#    lmexpdir=exp/${lmexpname}
#    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# reaults: cat log/steps_new_9.log
fi


# case 4:/home4/backup/haoxiang215/outdata/BWE_DNN_notargetcmvn/results/*.wav
# case 5:/home4/backup/haoxiang215/outdata/BWE_Spline/*.wav
# case 6:/home/haoxiang215/workspace/audio/data/reformat_data/src/*.wav.pr.wav
# case 7:/home/haoxiang215/workspace/audio/data/reformat_data/src1/*.wav.pr.wav
# case 8:/home/haoxiang215/workspace/audio/data/reformat_data/src6/*.wav.pr.wav
# case 9:/home/haoxiang215/workspace/audio/data/reformat_data/src7_concat/*.wav.pr.wav

# make wav
if [ ! -z $step10 ];then
   data_4=data/BWE_DNN_notargetcmvn
   srcdata_4=/home4/backup/haoxiang215/outdata/BWE_DNN_notargetcmvn/results
   [ -d $data_4 ] || mkdir -p $data_4
      find $srcdata_4 -name "*.wav" | \
      source-md/egs/haoxiang_research/make_wavscp_1.pl | sort -n -k 1 -t " " > $data_4/wav.scp
   data_5=data/BWE_Spline
   srcdata_5=/home4/backup/haoxiang215/outdata/BWE_Spline
   [ -d $data_5 ] || mkdir -p $data_5
      find $srcdata_5 -name "*.wav" | \
      source-md/egs/haoxiang_research/make_wavscp_1.pl | sort -n -k 1 -t " " > $data_5/wav.scp

   data_6=data/src
   srcdata_6=/home/haoxiang215/workspace/audio/data/reformat_data/src
   [ -d $data_6 ] || mkdir -p $data_6

      find $srcdata_6 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_6/wav.scp
   data_7=data/src1
   srcdata_7=/home/haoxiang215/workspace/audio/data/reformat_data/src1
   [ -d $data_7 ] || mkdir -p $data_7

      find $srcdata_7 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_7/wav.scp
   data_8=data/src6
   srcdata_8=/home/haoxiang215/workspace/audio/data/reformat_data/src6
   [ -d $data_8 ] || mkdir -p $data_8

      find $srcdata_8 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_8/wav.scp
   data_9=data/src7_concat
   srcdata_9=/home/haoxiang215/workspace/audio/data/reformat_data/src7_concat
   [ -d $data_9 ] || mkdir -p $data_9

      find $srcdata_9 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_9/wav.scp

fi
data_4=data/BWE_DNN_notargetcmvn
data_5=data/BWE_Spline
data_6=data/src
data_7=data/src1
data_8=data/src6
data_9=data/src7_concat
if [ ! -z $step11 ];then
   #  make utt2spk spk2utt
   for data in $data_4 $data_5 $data_6 $data_7 $data_8 $data_9;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step12 ];then
   # make text
   for data in $data_4 $data_5 $data_6 $data_7 $data_8 $data_9;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi

# make input data format for recognition
recog_set="BWE_DNN_notargetcmvn BWE_Spline src src1 src6 src7_concat"
# make features for dev_new
if [ ! -z $step13 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
if [ ! -z $step14 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format 
if [ ! -z $step15 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
# sbatch -o log/steps16.log source-md/egs/haoxiang_research/run_1a.sh --steps 16
if [ ! -z $step16 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    
        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# results: cat log/steps16_1.log 
fi

#### third experiment
# case 10:/home/haoxiang215/workspace/audio/data/reformat_data/2srcf47/*.wav.pr.wav
# case 11:/home/haoxiang215/workspace/audio/data/reformat_data/2srcf50/*.wav.pr.wav
# case 12:/home/haoxiang215/workspace/audio/data/reformat_data/2srcf70/*.wav.pr.wav
# case 13:/home/haoxiang215/workspace/audio/data/reformat_data/2srcf74/*.wav.pr.wav
# case 14:/home/haoxiang215/workspace/audio/data/reformat_data/2srcf76/*.wav.pr.wav
if [ ! -z $step17 ];then
   data_10=data/2srcf47
   srcdata_10=/home/haoxiang215/workspace/audio/data/reformat_data/2srcf47
   [ -d $data_10 ] || mkdir -p $data_10

      find $srcdata_10 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_10/wav.scp
   data_11=data/2srcf50
   srcdata_11=/home/haoxiang215/workspace/audio/data/reformat_data/2srcf50
   [ -d $data_11 ] || mkdir -p $data_11

      find $srcdata_11 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_11/wav.scp
   data_12=data/2srcf70
   srcdata_12=/home/haoxiang215/workspace/audio/data/reformat_data/2srcf70
   [ -d $data_12 ] || mkdir -p $data_12

      find $srcdata_12 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_12/wav.scp
   data_13=data/2srcf74
   srcdata_13=/home/haoxiang215/workspace/audio/data/reformat_data/2srcf74
   [ -d $data_13 ] || mkdir -p $data_13

      find $srcdata_13 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_13/wav.scp

   data_14=data/2srcf76
   srcdata_14=/home/haoxiang215/workspace/audio/data/reformat_data/2srcf76
   [ -d $data_14 ] || mkdir -p $data_14

      find $srcdata_14 -name "*.wav.pr.wav" | \
      source-md/egs/haoxiang_research/make_wavscp.pl | sort -n -k 1 -t " "> $data_14/wav.scp
fi

data_10=data/2srcf47
data_11=data/2srcf50
data_12=data/2srcf70
data_13=data/2srcf74
data_14=data/2srcf76
if [ ! -z $step18 ];then
   #  make utt2spk spk2utt
   for data in $data_10 $data_11 $data_12 $data_13 $data_14 ;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step19 ];then
   # make text
   for data in $data_10 $data_11 $data_12 $data_13 $data_14 ;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi
# make input data format for recognition
recog_set="2srcf47 2srcf50 2srcf70 2srcf74 2srcf76"
# make features for dev_new
if [ ! -z $step20 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
if [ ! -z $step21 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format 
if [ ! -z $step22 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set="2srcf50 2srcf70 2srcf74 2srcf76"
# sbatch -o log/steps17-23.log source-md/egs/haoxiang_research/run_1a.sh --steps 17-23
if [ ! -z $step23 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do

        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# results: 
fi
#data_15: raw 16k, /home/haoxiang215/clean_testset_wav2
data_15=clean_testset_wav2
if [ ! -z $step24 ];then
   data_15=data/clean_testset_wav2
   srcdata_15=/home/haoxiang215/clean_testset_wav2
   [ -d $data_15 ] || mkdir -p $data_15

     find $srcdata_15 -name "*.wav" | \
      source-md/egs/haoxiang_research/make_wavscp_1.pl | sort -n -k 1 -t " " > $data_15/wav.scp

fi

data_15=data/clean_testset_wav2
if [ ! -z $step25 ];then
   #  make utt2spk spk2utt
   for data in  $data_15 ;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step26 ];then
   # make text
   for data in $data_15 ;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi
# make input data format for recognition
recog_set="clean_testset_wav2"
# make features for dev_new
if [ ! -z $step27 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
# dump features 
if [ ! -z $step28 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format 
if [ ! -z $step29 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set="clean_testset_wav2"
# sbatch -o log/steps17-23.log source-md/egs/haoxiang_research/run_1a.sh --steps 17-23
if [ ! -z $step30 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do

        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# results: 
fi

# data_17;/home/haoxiang215/2clean_testset_wav_re16k
#data_17=2clean_testset_wav_re16k
if [ ! -z $step31 ];then
   data_17=data/2clean_testset_wav_re16k
   srcdata_17=/home/haoxiang215/2clean_testset_wav_re16k
   [ -d $data_17 ] || mkdir -p $data_17

     find $srcdata_17 -name "*.wav" | \
      source-md/egs/haoxiang_research/make_wavscp_1.pl | sort -n -k 1 -t " " > $data_17/wav.scp


fi

data_17=data/2clean_testset_wav_re16k
if [ ! -z $step32 ];then
   #  make utt2spk spk2utt
   for data in   $data_17 ;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step33 ];then
   # make text
   for data in $data_17 ;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi
# make input data format for recognition
recog_set=" 2clean_testset_wav_re16k"
# make features for dev_new
if [ ! -z $step34 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
# dump features 
if [ ! -z $step35 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format 
if [ ! -z $step36 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set=" 2clean_testset_wav_re16k"
# sbatch -o log/steps17-23.log source-md/egs/haoxiang_research/run_1a.sh --steps 17-23
# sbatch -o log/steps17-23.log source-md/egs/haoxiang_research/run_1a.sh --steps 17-23
if [ ! -z $step37 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do

        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# results: 
fi


if [ ! -z $step38 ];then
   data_18=data/clean_testset_wav_re16k2
   srcdata_18=/home/haoxiang215/clean_testset_wav_re16k2
   [ -d $data_18 ] || mkdir -p $data_18

     find $srcdata_18 -name "*.wav" | \
      source-md/egs/haoxiang_research/make_wavscp_1.pl | sort -n -k 1 -t " " > $data_18/wav.scp
   

fi
data_18=data/clean_testset_wav_re16k2
if [ ! -z $step39 ];then
   #  make utt2spk spk2utt
   for data in   $data_18 ;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step40 ];then
   # make text
   for data in $data_18 ;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi
# make input data format for recognition
recog_set=" clean_testset_wav_re16k2"
# make features for dev_new
if [ ! -z $step41 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
# dump features 
if [ ! -z $step42 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format 
if [ ! -z $step43 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set=" clean_testset_wav_re16k2"
# sbatch -o log/steps17-23.log source-md/egs/haoxiang_research/run_1a.sh --steps 17-23
# sbatch -o log/steps17-23.log source-md/egs/haoxiang_research/run_1a.sh --steps 17-23
if [ ! -z $step44 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do

        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# results: 
fi

if [ ! -z $step45 ];then
   data_19=data/clean_testset_wav_re16k3
   srcdata_19=/home/haoxiang215/clean_testset_wav_re16k3
   [ -d $data_19 ] || mkdir -p $data_19

     find $srcdata_19 -name "*.wav" | \
      source-md/egs/haoxiang_research/make_wavscp_1.pl | sort -n -k 1 -t " " > $data_19/wav.scp


fi
data_19=data/clean_testset_wav_re16k3
if [ ! -z $step46 ];then
   #  make utt2spk spk2utt
   for data in   $data_19 ;do
      cat $data/wav.scp | awk '{print $1, $1;}' | sort> $data/utt2spk
      utils/utt2spk_to_spk2utt.pl $data/utt2spk > $data/spk2utt
   done
fi
if [ ! -z $step47 ];then
   # make text
   for data in $data_19 ;do
      find /home4/backup/haoxiang215/data/Valentini-Botinhao/testset_txt/ -name "*.txt"> $data/testset_list
      cat $data/testset_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $data/text_scp_trainset.txt

      # corvet text_scp to kaldi text
      source-md/egs/nana_research/make_text.py $data/text_scp_trainset.txt $data/pre_text
      # remove punctuation 
      sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $data/pre_text | sort >$data/text_1
      # covert lower to upper
      source-md/egs/haoxiang_research/upper_text.py $data/text_1 > $data/text
   done
fi
# make input data format for recognition
recog_set=" clean_testset_wav_re16k3"

# make features for dev_new
if [ ! -z $step48 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in $recog_set ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            data/${x} exp/make_fbank/${x} ${fbankdir}
    done
fi
train_set=train_960
# dump features 
# dump features 
if [ ! -z $step49 ];then
   for rtask in $recog_set; do
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            data/${rtask}/feats.scp data/${train_set}/cmvn.ark exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format 
if [ ! -z $step50 ];then
    for rtask in $recog_set; do
       feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set=" clean_testset_wav_re16k3"
if [ ! -z $step51 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_pytorch_transformer_large_ngpu4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_large.yaml
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    #if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
       # Average ASR model
       expdir=exp/train_960_pytorch_train_pytorch_transformer.v1_aheads8_batch-bins15000000_specaug
       recog_model=model.val5.avg.best
       # lm model
       lmexpdir=exp/irielm.ep11.last5.avg
       lang_model=rnnlm.model.best
    #fi
    nj=32

    pids=() # initialize pids
    for rtask in ${recog_set}; do

        decode_dir=decode_${rtask}_${recog_model}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        # set batchsize 0 to disable batch decoding
        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --preprocess-conf ${preprocess_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model} \
            --rnnlm ${lmexpdir}/${lang_model}

        score_sclite.sh --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}
    pids+=($!) # store background pids
     done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# results: 
fi

