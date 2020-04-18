#!/bin/bash

##espnet version:0.5.2
#input feature:80-dim fbank(add pitch)
# model unit is sentencepiece(e.g.:bpe).

. path_v1.sh
. cmd.sh
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
nbpe=3000
bpemode=bpe
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

 # Set bash to 'debug' mode, it will exit on :
 # -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
 set -e
# set -u
 set -o pipefail
tgtdir=run_1b
train_set=train_trn
train_dev=train_dev
recog_set="test_set dev_sge dev_man"
feat_tr_dir=${tgtdir}/${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${tgtdir}/${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

if [ ! -z $step01 ];then
   #
   
   sed  -e "s,/mipirepo,/home4/zhb502/w2019/projects," $tgtdir/data/train/wav.scp >$tgtdir/data/train/wav_1.scp
   rm -rf $tgtdir/data/train/wav.scp
   mv $tgtdir/data/train/wav_1.scp  $tgtdir/data/train/wav.scp  
   
   utils/validate_data_dir.sh --no-feats $tgtdir/data/train || exit 1;
   sed  -e "s,/mipirepo,/home4/zhb502/w2019/projects," $tgtdir/data/test_set/wav.scp >$tgtdir/data/test_set/wav_1.scp
   rm -rf $tgtdir/data/test_set/wav.scp
   mv $tgtdir/data/test_set/wav_1.scp  $tgtdir/data/test_set/wav.scp 
   #rm -rf $tgtdir/data/test_set/text_character_1
   utils/validate_data_dir.sh --no-feats $tgtdir/data/test_set || exit 1;
   
fi

src_datadir=/home4/md510/w2018/data/seame
if [ ! -z $step02 ];then
    echo "stage 1: Data preparation"
    for part in dev_man dev_sge; do
        # use underscore-separated names in data directories.
        #local/data_prep.sh ${src_datadir}/LibriSpeech/${part} data/$(echo ${part} | sed s/-/_/g)
        utils/copy_data_dir.sh ${src_datadir}/${part} $tgtdir/data/${part}|| exit 1;
        utils/validate_data_dir.sh --no-feats $tgtdir/data/${part} || exit 1;
    done

fi
if [ ! -z $step03 ];then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
    for x in dev_man dev_sge test_set train; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi

if [ ! -z $step04 ]; then
     #utils/subset_data_dir_tr_cv.sh $tgtdir/data/train $tgtdir/data/${train_set}_org $tgtdir/data/${train_dev}_org || exit 1;
     # remove utt having more than 3000 frames
     # remove utt having more than 400 characters
     #remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${tgtdir}/data/${train_set}_org $tgtdir/data/${train_set}_cut
     #remove_longshortdata.sh --maxframes 3000 --maxchars 400 ${tgtdir}/data/${train_dev}_org $tgtdir/data/${train_dev}_cut

     # make train_dev from train
     echo "LOG:::make a train_dev and train_trn from train set"
     # train has  #utt 178246,  540 speakers
     
     # train_trn : train_dev =9:1
     utils/subset_data_dir_tr_cv.sh $tgtdir/data/train $tgtdir/data/train_trn $tgtdir/data/${train_dev} || exit 1;

     fbankdir=fbank
     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 32 --write_utt2num_frames true \
        ${tgtdir}/data/${train_set} ${tgtdir}/exp/make_fbank/${train_set} ${tgtdir}/${fbankdir}
     utils/fix_data_dir.sh ${tgtdir}/data/${train_set}
     # train_sp has    utterances 
     echo "LOG:::make speed-perturbed for train_trn set done"
fi

if [ ! -z $step05 ]; then
    #utils/spk2utt_to_utt2spk.pl data/train/spk2utt  > data/train/utt2spk
    # compute global CMVN for ${train_set} ,in order to dump . ${train_dev} and ${recog_set} are not required.
    echo "LOG:::compute global CMVN "
    compute-cmvn-stats scp:${tgtdir}/data/${train_set}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark
    # compute-cmvn-stats scp:${tgtdir}/data/${train_dev}_cut/feats.scp ${tgtdir}/data/${train_dev}_cut/cmvn.ark
    echo "LOG:::cmvn done"
fi

if [ ! -z $step06 ]; then
    echo " LOG:::dump features"
    # dump features
    dump.sh --cmd "$train_cmd" --nj 30 --do_delta $do_delta \
        ${tgtdir}/data/${train_set}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/train ${feat_tr_dir}
   # utils/spk2utt_to_utt2spk.pl data/train/spk2utt  > data/train/utt2spk
    dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
       ${tgtdir}/data/${train_dev}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
   echo " LOG:::dump feature done"
fi

dict=${tgtdir}/data/lang_char/${train_set}_${bpemode}${nbpe}_units.txt
bpemodel=${tgtdir}/data/lang_char/${train_set}_${bpemode}${nbpe}
echo "dictionary: ${dict}"
#nlsyms=${tgtdir}/data/lang_letter/non_lang_syms.txt  # in order to make a dictionary

if [ ! -z $step07 ];then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "LOG:::stage 6: Dictionary  Preparation"
    mkdir -p ${tgtdir}/data/lang_char/

    echo "LOG:::in order to make a dictionary"
    echo "<unk> 1">${dict} #<unk> must be 1, 0 will be used for "blank" in CTC.
    cut -f 2- -d" " ${tgtdir}/data/train/text>${tgtdir}/data/lang_char/input.txt
    #cut -f 2- ${tgtdir}/data/train/text>${tgtdir}/data/lang_char/input.txt
    spm_train --input=${tgtdir}/data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece<${tgtdir}/data/lang_char/input.txt | tr ' ' '\n'| sort | uniq | awk '{print $0 " " NR+1}'>>${dict}
    wc -l ${dict}
    echo "LOG::: Dictionary  Preparation done"
fi

if [ ! -z $step08 ]; then
   echo " LOG:::make json labels"
   echo "${tgtdir}/data/train/utt2spk number=`cat ${tgtdir}/data/train/utt2spk | wc -l`"
   # make json labels
   data2json.sh --feat ${feat_tr_dir}/feats.scp --bpecode ${bpemodel}.model \
         $tgtdir/data/${train_set} ${dict} > ${feat_tr_dir}/data_${bpemode}${nbpe}.json
   data2json.sh --feat ${feat_dt_dir}/feats.scp --bpecode ${bpemodel}.model \
         $tgtdir/data/${train_dev} ${dict} > ${feat_dt_dir}/data_${bpemode}${nbpe}.json
   for rtask in ${recog_set}; do
       feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done
   echo " LOG:::make json labels done"
fi

###add character lm
if [ ! -z $step09 ]; then
   echo "stage 09: character LM Preparation"   
   lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}
   lmdict=${dict}
   mkdir -p ${lmdatadir}
    cat $tgtdir/data/train/text | \
      spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " $tgtdir/data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/valid.txt
   echo "stage 09: character LM Preparation done"
fi
# use only 1 gpu
if [ ! -z $step10 ];then
    lm_config=conf/lm_default.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${lm_ngpu} \
        --backend ${backend} \
        --verbose 0 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/$lmexpdir \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ ! -z $step11 ];then
    train_config=conf/train_hkust_conv1d_statistic.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yam
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v1.sh ] && . ./path_v1.sh 
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

# make prepared dev
[ -d ${tgtdir}/data/dev ] || mkdir -p ${tgtdir}/data/dev
if [ ! -z $step12 ];then
   # /home/md510/w2019/project/espnet-recipe/asru2019/run_1a/data/dev/data/category/G0476/session01/T0392G0476_S01010001.wav
   cp -r /home4/md510/w2019a/espnet-recipe/asru2019/run_1e_statisitc/data/dev/* ${tgtdir}/data/dev
   sed  -e "s,/home/md510/w2019/project/espnet-recipe/asru2019/run_1a,/home4/md510/w2019a/espnet-recipe/asru2019/run_1e_statisitc," $tgtdir/data/dev/wav.scp >$tgtdir/data/dev/wav_1.scp
   head -n 5 $tgtdir/data/dev/wav_1.scp
   mv $tgtdir/data/dev/wav_1.scp $tgtdir/data/dev/wav.scp
fi

# make features for dev
if [ ! -z $step13 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in dev ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi
# dump features for dev
if [ ! -z $step14 ];then
   for rtask in dev; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format for dev
if [ ! -z $step15 ];then
    for rtask in dev; do
       feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi

if [ ! -z $step16 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then

        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
    for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_conv1d_statistic.yaml
# preprocess_config=conf/specaug.yaml
# ecode_config=conf/decode_hkust.yam
# lm_config=conf/lm_default.yaml

# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# dev    12.59
# dev_CH 9.4
# dev_EN 30.8
fi

if [ ! -z $step17 ];then
    train_config=conf/train_hkust_conv1d_statistic_small.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v1.sh ] && . ./path_v1.sh
    cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step18 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_small.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then

        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
    for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_conv1d_statistic_small.yaml
# preprocess_config=conf/specaug.yaml
# ecode_config=conf/decode_hkust.yam
# lm_config=conf/lm_default.yaml

# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
#         CH: %CER 12.54 [ 21727 / 173296, 840 ins, 1434 del, 19453 sub ]
#         EN: %WER 39.21 [ 11910 / 30375, 826 ins, 606 del, 10478 sub ]
#         MIX: %MER 16.52 [ 33637 / 203671, 1666 ins, 2040 del, 29931 sub ]
 
fi

# replace positionwisefeedforward layer with conv1d in decoer part of transformer.
# error log ;  
if [ ! -z $step19 ];then
    train_config=conf/train_hkust_enocoder_conv1d_1_decoder_conv1d_1_embed_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node05"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step20 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_1_decoder_conv1d_1_embed_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
    for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_enocoder_conv1d_decoder_covn1d.yaml
# preprocess_config=conf/specaug.yaml
# ecode_config=conf/decode_hkust.yam
# lm_config=conf/lm_default.yaml

# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

fi






# make kaldi formate for ASRUn20 set

if [ ! -z $step21 ];then
   # make wav.scp
   [ -d $tgtdir/data/dev_new ] || mkdir -p $tgtdir/data/dev_new
   cp -r data/ASRUn20/wav.scp $tgtdir/data/dev_new/ 
   sed -e "s,/home/backup_nfs/data-ASR/ASRUTestData/mix/n100小时中英混读手机采集语音数据/./category,/home4/md510/w2019a/espnet-recipe/asru2019/data/ASRUn20/data,"  $tgtdir/data/dev_new/wav.scp > $tgtdir/data/dev_new/wav_1.scp
   head -n 5 $tgtdir/data/dev_new/wav_1.scp
   cat $tgtdir/data/dev_new/wav_1.scp | sort > $tgtdir/data/dev_new/wav_2.scp
   mv $tgtdir/data/dev_new/wav_2.scp $tgtdir/data/dev_new/wav.scp
   
fi
if [ ! -z $step22 ];then
   # make text
   srcdata=/home4/md510/w2019a/espnet-recipe/asru2019/data/ASRUn20/data
   find $srcdata/*/*/  -name "*.txt"  > $tgtdir/data/dev_new/text_1
   source-md/egs/asru2019/make_text_scp_for_dev_new.py $tgtdir/data/dev_new/text_1 > $tgtdir/data/dev_new/text_1_scp 
   source-md/egs/asru2019/make_text_for_dev_new.py $tgtdir/data/dev_new/text_1_scp $tgtdir/data/dev_new/text_2 
   # covert word to character
   [ -f ./path_v1.sh ] && . ./path_v1.sh
   cat $tgtdir/data/dev_new/text_2 | PYTHONIOENCODING=utf-8  ./source-scripts/egs/mandarin/update-april-03-2017-with-pruned-lexicon/segment-chinese-text.py  --do-character-segmentation >$tgtdir/data/dev_new/text_character_1
   head -n 10 $tgtdir/data/dev_new/text_character_1
   # remove specify symble  for normoalize text
   source-md/egs/asru2019/normalize_dev_text.py $tgtdir/data/dev_new/text_character_1 > $tgtdir/data/dev_new/text_character_2
   head -n 10 $tgtdir/data/dev_new/text_character_2
   mv $tgtdir/data/dev_new/text_character_2 $tgtdir/data/dev_new/text_3
   cat $tgtdir/data/dev_new/text_3 | sort >$tgtdir/data/dev_new/text

fi

if [ ! -z $step23 ];then
   # make utt2spk spk2utt
   cp -r  data/ASRUn20/utt2spk  run_1b/data/dev_new/
   cat run_1b/data/dev_new/utt2spk | sort > run_1b/data/dev_new/utt2spk_1
   mv run_1b/data/dev_new/utt2spk_1 run_1b/data/dev_new/utt2spk
   utils/utt2spk_to_spk2utt.pl run_1b/data/dev_new/utt2spk > run_1b/data/dev_new/spk2utt 
   # final data 
   [ -d data/dev_new ] || mkdir -p data/dev_new
   cp -r run_1b/data/dev_new/text data/dev_new/
   cp -r  run_1b/data/dev_new/spk2utt data/dev_new/
   cp -r run_1b/data/dev_new/utt2spk data/dev_new/
   cp -r run_1b/data/dev_new/wav.scp data/dev_new/
fi

# make features for dev_new
if [ ! -z $step25 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in dev_new ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi

# dump features for dev_new
if [ ! -z $step26 ];then
   for rtask in dev_new; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format for dev_new
if [ ! -z $step27 ];then
    for rtask in dev_new; do
       feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi



if [ ! -z $step30 ];then
    train_config=conf/train_hkust_enocoder_conv1d_decoder_covn1d_v1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node05"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=3 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ! -z $step31 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_decoder_covn1d_v1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_enocoder_conv1d_decoder_covn1d_v1.yaml
# preprocess_config=conf/specaug.yaml
# ecode_config=conf/decode_hkust.yam
# lm_config=conf/lm_default.yaml

# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
                             
# dev
# CH: %CER 9.90 [ 17151 / 173296, 522 ins, 1018 del, 15611 sub ]
# EN: %WER 32.38 [ 9836 / 30375, 611 ins, 510 del, 8715 sub ]
# MIX: %MER 13.25 [ 26987 / 203671, 1133 ins, 1528 del, 24326 sub ]
# Result:9.90%,32.38%,13.25%
# k40m single gpu
# elapsed time 181009 seconds

fi
if [ ! -z $step32 ];then
    train_config=conf/train_hkust_enocoder_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node05"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ! -z $step33 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# dev
#  CH: %CER 9.89 [ 17136 / 173296, 537 ins, 986 del, 15613 sub ]
#  EN: %WER 32.30 [ 9810 / 30375, 593 ins, 526 del, 8691 sub ]
#  MIX: %MER 13.23 [ 26946 / 203671, 1130 ins, 1512 del, 24304 sub ]
#  Result:9.89%,32.30%,13.23%
# k40m :single gpu
# elapsed time 130623 seconds 
fi

if [ ! -z $step34 ];then
    train_config=conf/train_hkust_enocoder_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node05"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=3 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step35 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# dev
# CH: %CER 9.41 [ 16308 / 173296, 534 ins, 912 del, 14862 sub ]
# EN: %WER 30.05 [ 9128 / 30375, 574 ins, 438 del, 8116 sub ]
#MIX: %MER 12.49 [ 25436 / 203671, 1108 ins, 1350 del, 22978 sub ]
#Result:9.41%,30.05%,12.49%
fi

if [ ! -z $step36 ];then
    train_config=conf/train_hkust_enocoder_conv1d_7_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node05"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=2 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
# custom :9312M single k40m
fi

if [ ! -z $step37 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_7_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}
    # model average realted (only for transformer)
    n_average=5                  # the number of ASR models to be averaged
    use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
    lm_n_average=6               # the number of languge models to be averaged
    use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
#        if ${use_lm_valbest_average}; then
#            lang_model=rnnlm.val${n_average}.avg.best
#            opt="--log ${expdir}/results/log"
#        else
#            lang_model=rnnlm.last${n_average}.avg.best
#        fi
#        average_checkpoints.py \
#            ${opt} \
#            --backend ${backend} \
#            --snapshots ${lmexpdir}/snapshot.ep.* \
#            --out ${lmexpdir}/${lang_model} \
#            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev ; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false


# dev
# CH: %CER 9.74 [ 16885 / 173296, 549 ins, 1002 del, 15334 sub ]
# EN: %WER 30.13 [ 9152 / 30375, 555 ins, 441 del, 8156 sub ]
# MIX: %MER 12.78 [ 26037 / 203671, 1104 ins, 1443 del, 23490 sub ]
# Result:9.74%,30.13%,12.78%

fi

if [ ! -z $step38 ];then
    train_config=conf/train_hkust_enocoder_conv1d-linear_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step39 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_conv1d-linear_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
        lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# dev
#  CH: %CER 9.66 [ 16748 / 173296, 604 ins, 924 del, 15220 sub ]
#  EN: %WER 31.28 [ 9500 / 30375, 595 ins, 474 del, 8431 sub ]
#MIX: %MER 12.89 [ 26248 / 203671, 1199 ins, 1398 del, 23651 sub ]
#Result:9.66%,31.28%,12.89%
fi


if [ ! -z $step40 ];then
    train_config=conf/train_hkust_enocoder_causal-conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step41 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_causal-conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
        lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

fi


if [ ! -z $step42 ];then
    train_config=conf/train_hkust_enocoder_linear_decoder_linear.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step43 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_linear_decoder_linear.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
        lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
         average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev
# CH: %CER 12.18 [ 21107 / 173296, 913 ins, 1260 del, 18934 sub ]
# EN: %WER 38.86 [ 11805 / 30375, 873 ins, 550 del, 10382 sub ]
#MIX: %MER 16.16 [ 32912 / 203671, 1786 ins, 1810 del, 29316 sub ]
#Result:12.18%,38.86%,16.16%
fi
if [ ! -z $step44 ];then
    train_config=conf/train_hkust_enocoder_conv1d_1_decoder_linear.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step45 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_conv1d_1_decoder_linear.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
        lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
         fi
         average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false


# dev
#  CH: %CER 12.02 [ 20827 / 173296, 846 ins, 1250 del, 18731 sub ]
# EN: %WER 38.26 [ 11622 / 30375, 834 ins, 557 del, 10231 sub ]
#MIX: %MER 15.93 [ 32449 / 203671, 1680 ins, 1807 del, 28962 sub ]
#Result:12.02%,38.26%,15.93%
fi


if [ ! -z $step46 ];then
    train_config=conf/train_hkust_enocoder_conv1d_1_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=2 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi
if [ ! -z $step47 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_conv1d_1_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
         # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
        lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
         fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false


# CH: %CER 12.51 [ 21674 / 173296, 877 ins, 1352 del, 19445 sub ]
# EN: %WER 39.26 [ 11925 / 30375, 857 ins, 583 del, 10485 sub ]
#MIX: %MER 16.50 [ 33599 / 203671, 1734 ins, 1935 del, 29930 sub ]
#Result:12.51%,39.26%,16.50%
fi

# in order to comapre conf/train_hkust_conv1d_statistic.yaml  in steps 16
if [ ! -z $step50 ];then
    train_config=conf/train_hkust_conv1d_without_statistic.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yam
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    cuda_cmd="slurm.pl --quiet --nodelist=node07"
     # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/$expdir \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --train-json ${feat_tr_dir}/data_${bpemode}${nbpe}.json \
        --valid-json ${feat_dt_dir}/data_${bpemode}${nbpe}.json
fi

if [ ! -z $step51 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_without_statistic.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_hkust.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}

    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then

        # Average ASR models
        if ${use_valbest_average}; then
            recog_model=model.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            recog_model=model.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${expdir}/results/snapshot.ep.* \
            --out ${expdir}/results/${recog_model} \
            --num ${n_average}

        # Average LM models
        if ${use_lm_valbest_average}; then
            lang_model=rnnlm.val${n_average}.avg.best
            opt="--log ${expdir}/results/log"
        else
            lang_model=rnnlm.last${n_average}.avg.best
            opt="--log"
        fi
        average_checkpoints.py \
            ${opt} \
            --backend ${backend} \
            --snapshots ${lmexpdir}/snapshot.ep.* \
            --out ${lmexpdir}/${lang_model} \
            --num ${lm_n_average}
    fi
    nj=32
    pids=() # initialize pids
     for rtask in dev; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data_${bpemode}${nbpe}.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --batchsize 0 \
            --recog-json ${feat_recog_dir}/split${nj}utt/data_${bpemode}${nbpe}.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            --rnnlm ${lmexpdir}/rnnlm.model.best

        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev
#  CH: %CER 9.60 [ 16631 / 173296, 641 ins, 854 del, 15136 sub ]
#  EN: %WER 31.26 [ 9496 / 30375, 696 ins, 420 del, 8380 sub ]
# MIX: %MER 12.83 [ 26127 / 203671, 1337 ins, 1274 del, 23516 sub ]
# Result:9.60%,31.26%,12.83%

fi
