#!/bin/bash

##espnet version:0.5.3
#input feature:80-dim fbank(add pitch)
# model unit is sentencepiece(e.g.:bpe).
#            english unit:41 lowercase English letters word pieces
#            mandarin unit:3918 single Chinese characters.  
#            <unk> ' - . 

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
tgtdir=run_1e_statisitc
train_set=train_trn_sp
train_dev=train_dev
recog_set="dev"
feat_tr_dir=${tgtdir}/${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${tgtdir}/${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

[ -d $tgtdir/data/cs200 ] || mkdir -p $tgtdir/data/cs200
# preprocess  cs200 
if [ ! -z $step01 ];then
   # process wav.scp
   cp -r /home4/md510/w2019a/espnet-recipe/asru2019/run_1b/data/cs200/* $tgtdir/data/cs200 
   sed  -e "s,/mipirepo,/home4/zhb502/w2019/projects," $tgtdir/data/cs200/wav.scp >$tgtdir/data/cs200/wav_1.scp
   rm -rf $tgtdir/data/cs200/wav.scp
   mv $tgtdir/data/cs200/wav_1.scp  $tgtdir/data/cs200/wav.scp
   # process text
   [ -f ./path_v1.sh ] && . ./path_v1.sh
  cat $tgtdir/data/cs200/text | PYTHONIOENCODING=utf-8  ./source-scripts/egs/mandarin/update-april-03-2017-with-pruned-lexicon/segment-chinese-text.py  --do-character-segmentation >$tgtdir/data/cs200/text_character_1
   head -n 10 $tgtdir/data/cs200/text_character_1
   mv $tgtdir/data/cs200/text_character_1 $tgtdir/data/cs200/text
   head -n 10 $tgtdir/data/cs200/text 
   utils/validate_data_dir.sh --no-feats $tgtdir/data/cs200 || exit 1;
fi


[ -d ${tgtdir}/data/dev ] || mkdir -p ${tgtdir}/data/dev
if [ ! -z $step02 ];then
   # /home/md510/w2019/project/espnet-recipe/asru2019/run_1a/data/dev/data/category/G0476/session01/T0392G0476_S01010001.wav
   sed  -e "s,/home/md510/w2019/project/espnet-recipe/asru2019/run_1a,/home4/md510/w2019a/espnet-recipe/asru2019/run_1e_statisitc," $tgtdir/data/dev/wav.scp >$tgtdir/data/dev/wav_1.scp
   head -n 5 $tgtdir/data/dev/wav_1.scp
   mv $tgtdir/data/dev/wav_1.scp $tgtdir/data/dev/wav.scp
fi


if [ ! -z $step03 ];then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
    for x in dev cs200; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi
if [ ! -z $step04 ]; then
     # make train_dev from train
     echo "LOG:::make a train_dev and train_trn from train set"
     # train_trn : train_dev =95:5
     utils/subset_data_dir_tr_cv.sh  --cv-spk-percent 5 $tgtdir/data/cs200 $tgtdir/data/train_trn $tgtdir/data/${train_dev} || exit 1;

     # speed-perturbed for train_trn set
     echo "LOG:::make speed-perturbed for train_trn set "
     utils/perturb_data_dir_speed.sh 0.9  ${tgtdir}/data/train_trn  ${tgtdir}/data/temp1
     utils/perturb_data_dir_speed.sh 1.0  ${tgtdir}/data/train_trn  ${tgtdir}/data/temp2
     utils/perturb_data_dir_speed.sh 1.1  ${tgtdir}/data/train_trn  ${tgtdir}/data/temp3
     utils/combine_data.sh --extra-files utt2uniq ${tgtdir}/data/${train_set} ${tgtdir}/data/temp1 ${tgtdir}/data/temp2 ${tgtdir}/data/temp3
     rm -r  ${tgtdir}/data/temp1  ${tgtdir}/data/temp2  ${tgtdir}/data/temp3
     fbankdir=fbank
     steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50 --write_utt2num_frames true \
        ${tgtdir}/data/${train_set} ${tgtdir}/exp/make_fbank/${train_set} ${tgtdir}/${fbankdir}
     utils/fix_data_dir.sh ${tgtdir}/data/${train_set}
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
    cut -f 2- -d" " ${tgtdir}/data/train_trn/text>${tgtdir}/data/lang_char/input.txt
    #cut -f 2- ${tgtdir}/data/train/text>${tgtdir}/data/lang_char/input.txt
    spm_train --input=${tgtdir}/data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece<${tgtdir}/data/lang_char/input.txt | tr ' ' '\n'| sort | uniq | awk '{print $0 " " NR+1}'>>${dict}
    wc -l ${dict}
    echo "LOG::: Dictionary  Preparation done"
fi

if [ ! -z $step08 ]; then
   echo " LOG:::make json labels"
   echo "${tgtdir}/data/{train_set}/utt2spk number=`cat ${tgtdir}/data/{train_set}/utt2spk | wc -l`"
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
   echo "stage 08: character LM Preparation"   
   lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}
   lmdict=${dict}
   mkdir -p ${lmdatadir}
    # this is a small error.
    cat  $tgtdir/data/train_trn/text | \ 
      spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " $tgtdir/data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/valid.txt
   echo "stage 08: character LM Preparation done"
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
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/$lmexpdir \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi
if [ ! -z $step11 ];then
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
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


if [ ! -z $step12 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    for rtask in $recog_set; do
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
# train_config=conf/train_hkust_conv1d_statistic_large.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# dev     8.26 
# dev CH  6.38
# dev EN  18.98

fi
# increasing decoder layers
if [ ! -z $step13 ];then
    train_config=conf/train_hkust_conv1d_statistic_large_v2.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
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

if [ ! -z $step14 ];then
   echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large_v2.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    for rtask in $recog_set; do
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
# train_config=conf/train_hkust_conv1d_statistic_large_v2.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# dev
#  mix   9.29 
#  CH    7.32
#  EN    20.55


fi
# use only 1 gpu
if [ ! -z $step15 ];then
    lm_config=conf/lm_default_v4.yaml
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
if [ ! -z $step16 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default_v4.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    for rtask in $recog_set; do
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
# train_config=conf/train_hkust_conv1d_statistic_large.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default_v4.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# CH: %CER 16.81 [ 29139 / 173296, 841 ins, 7978 del, 20320 sub ]
# EN: %WER 41.98 [ 12752 / 30375, 645 ins, 1792 del, 10315 sub ]
#MIX: %MER 20.57 [ 41891 / 203671, 1486 ins, 9770 del, 30635 sub ]
#Result:16.81%,41.98%,20.57%
 

fi

# increasing linear units of encoder and decoder.
if [ ! -z $step17 ];then
    train_config=conf/train_hkust_conv1d_statistic_large_v5.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
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
    train_config=conf/train_hkust_conv1d_statistic_large_v5.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev dev_new"
    for rtask in $recog_set; do
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
# train_config=conf/train_hkust_conv1d_statistic_large_v5.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
#dev_new
#CH: %CER 12.09 [ 21870 / 180949, 617 ins, 989 del, 20264 sub ]
# EN: %WER 34.68 [ 7772 / 22413, 450 ins, 386 del, 6936 sub ]
#MIX: %MER 14.58 [ 29642 / 203362, 1067 ins, 1375 del, 27200 sub ]
#Result:12.09%,34.68%,14.58%


# dev
#CH: %CER 7.10 [ 12304 / 173296, 492 ins, 739 del, 11073 sub ]
# EN: %WER 20.52 [ 6232 / 30375, 419 ins, 305 del, 5508 sub ]
#MIX: %MER 9.10 [ 18536 / 203671, 911 ins, 1044 del, 16581 sub ]
#Result:7.10%,20.52%,9.10%

fi


if [ ! -z $step19 ];then
   [ -d $tgtdir/data/dev_new ] || mkdir -p $tgtdir/data/dev_new 
   cp -r data/dev_new/* $tgtdir/data/dev_new/
fi 

# make features for dev_new
if [ ! -z $step20 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in dev_new ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi

# dump features for dev_new
if [ ! -z $step21 ];then
   for rtask in dev_new; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format for dev_new
if [ ! -z $step22 ];then
    for rtask in dev_new; do
       feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set="dev_new"
if [ ! -z $step23 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new"
    for rtask in $recog_set; do
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
# train_config=conf/train_hkust_conv1d_statistic_large.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
# dev_new
#  CH: %CER 10.94 [ 19789 / 180949, 518 ins, 896 del, 18375 sub ]
#  EN: %WER 32.81 [ 7354 / 22413, 423 ins, 375 del, 6556 sub ]
#  MIX: %MER 13.35 [ 27143 / 203362, 941 ins, 1271 del, 24931 sub ]
#  Result:10.94%,32.81%,13.35%


fi

recog_set="dev_new"
if [ ! -z $step24 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new"
    for rtask in $recog_set; do
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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_conv1d_statistic_large.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# dev_new
# CH: %CER 10.92 [ 19767 / 180949, 524 ins, 890 del, 18353 sub ]
# EN: %WER 32.53 [ 7290 / 22413, 419 ins, 368 del, 6503 sub ]
# MIX: %MER 13.30 [ 27057 / 203362, 943 ins, 1258 del, 24856 sub ]
# Result:10.92%,32.53%,13.30%


fi

if [ ! -z $step25 ];then
    train_config=conf/train_hkust_enocoder_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
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
#recog_set="dev_new"
if [ ! -z $step26 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do
    
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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# custom;6013M, single k40m 
# dev_new
# CH: %CER 11.53 [ 20871 / 180949, 516 ins, 1010 del, 19345 sub ]
# EN: %WER 33.97 [ 7614 / 22413, 408 ins, 404 del, 6802 sub ]
# MIX: %MER 14.01 [ 28485 / 203362, 924 ins, 1414 del, 26147 sub ]
# Result:11.53%,33.97%,14.01%
# dev
# CH: %CER 7.02 [ 12167 / 173296, 408 ins, 728 del, 11031 sub ]
#  EN: %WER 20.36 [ 6184 / 30375, 419 ins, 310 del, 5455 sub ]
# MIX: %MER 9.01 [ 18351 / 203671, 827 ins, 1038 del, 16486 sub ]
# Result:7.02%,20.36%,9.01%

fi

if [ ! -z $step27 ];then
    train_config=conf/train_hkust_enocoder_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
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
if [ ! -z $step28 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# customs :6020M, single k40m 
# dev_new
#  CH: %CER 11.30 [ 20447 / 180949, 549 ins, 929 del, 18969 sub ]
#  EN: %WER 33.75 [ 7565 / 22413, 422 ins, 414 del, 6729 sub ]
# MIX: %MER 13.77 [ 28012 / 203362, 971 ins, 1343 del, 25698 sub ]
# Result:11.30%,33.75%,13.77% 
# dev
# CH: %CER 6.68 [ 11578 / 173296, 441 ins, 678 del, 10459 sub ]
# EN: %WER 20.17 [ 6128 / 30375, 360 ins, 337 del, 5431 sub ]
#MIX: %MER 8.69 [ 17706 / 203671, 801 ins, 1015 del, 15890 sub ]
#Result:6.68%,20.17%,8.69%

fi

if [ ! -z $step29 ];then
    train_config=conf/train_hkust_enocoder_20_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
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
fi
# next run
if [ ! -z $step30 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_20_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# single k40m 8987MiB 
# dev_new
#  CH: %CER 10.87 [ 19677 / 180949, 523 ins, 901 del, 18253 sub ]
# EN: %WER 33.20 [ 7442 / 22413, 406 ins, 416 del, 6620 sub ]
#MIX: %MER 13.34 [ 27119 / 203362, 929 ins, 1317 del, 24873 sub ]
#Result:10.87%,33.20%,13.34%

# dev
#  CH: %CER 6.49 [ 11249 / 173296, 446 ins, 656 del, 10147 sub ]
# EN: %WER 19.21 [ 5836 / 30375, 346 ins, 329 del, 5161 sub ]
#MIX: %MER 8.39 [ 17085 / 203671, 792 ins, 985 del, 15308 sub ]
#Result:6.49%,19.21%,8.39%
fi


if [ ! -z $step35 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev"
    for rtask in $recog_set; do
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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_conv1d_statistic_large.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.

# dev
# CH: %CER 6.38 [ 11049 / 173296, 441 ins, 659 del, 9949 sub ]
# EN: %WER 18.87 [ 5732 / 30375, 362 ins, 277 del, 5093 sub ]
#MIX: %MER 8.24 [ 16781 / 203671, 803 ins, 936 del, 15042 sub ]
#Result:6.38%,18.87%,8.24%

# dev_new
# CH: %CER 10.92 [ 19767 / 180949, 524 ins, 890 del, 18353 sub ]
# EN: %WER 32.53 [ 7290 / 22413, 419 ins, 368 del, 6503 sub ]
# MIX: %MER 13.30 [ 27057 / 203362, 943 ins, 1258 del, 24856 sub ]
# Result:10.92%,32.53%,13.30%


fi







# make prepared final official test set (e.g:/home4/md510/w2019a/espnet-recipe/asru2019/data/release) 
if [ ! -z $step50 ];then
   #make wav.scp
   [ -d $tgtdir/data/release ] || mkdir -p $tgtdir/data/release
   find /home4/md510/w2019a/espnet-recipe/asru2019/data/release/ -name "*.wav" > $tgtdir/data/release/wav_list
   cat $tgtdir/data/release/wav_list | perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";' | sort > $tgtdir/data/release/wav.scp
   head -n 5 $tgtdir/data/release/wav.scp

   #  make utt2spk spk2utt
   cat $tgtdir/data/release/wav.scp | awk '{print $1, $1;}' > $tgtdir/data/release/utt2spk
   utils/utt2spk_to_spk2utt.pl $tgtdir/data/release/utt2spk > $tgtdir/data/release/spk2utt


fi

if [ ! -z $step51 ];then
   # make features for release
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in release ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi
recog_set="release"
if [ ! -z $step52 ];then

   for rtask in ${recog_set}; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
   echo " LOG:::dump feature done"
fi
if [ ! -z $step53 ];then

   for rtask in ${recog_set}; do
       feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
       source-md/egs/asru2019/data2json_for_release.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done
   echo " LOG:::make json labels done"
fi

if [ ! -z $step54 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="release"
    for rtask in $recog_set; do
    
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
            --rnnlm ${lmexpdir}/${lang_model}


        #source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}

    
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# config:
# Time-consuming training model:
# data :speed-perturbed
# train_config=conf/train_hkust_conv1d_statistic_large.yaml
# preprocess_config=conf/specaug.yaml
# decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
# lm_config=conf/lm_default.yaml
# model average realted (only for transformer)
#n_average=5                  # the number of ASR models to be averaged
#use_valbest_average=true     # if true, the validation `n_average`-best ASR models will be averaged.
                             # if false, the last `n_average` ASR models will be averaged.
#lm_n_average=6               # the number of languge models to be averaged
#use_lm_valbest_average=false # if true, the validation `lm_n_average`-best language models will be averaged.
                             # if false, the last `lm_n_average` language models will be averaged.
# note:::::::::final offer this results
fi
# precess recognition results
if [ ! -z $step55 ];then
  train_config=conf/train_hkust_conv1d_statistic_large.yaml
  decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
  lm_config=conf/lm_default.yaml
  expname=${train_set}_${backend}_$(basename ${train_config%.*})
  expdir=$tgtdir/exp/${expname}
   ###add character lm
  if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
  fi
  recog_set="release"
  for rtask in $recog_set; do

   decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
   source-md/egs/asru2019/deal_with_recog_result_for_release.sh --steps 1-3 --bpe ${nbpe} --bpemodel ${bpemodel}.model  ${expdir}/${decode_dir} ${dict}  
  done
fi 



if [ ! -z $step56 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="release"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        #source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# custom;6013M, single k40m 
fi
# precess recognition results
if [ ! -z $step57 ];then
  train_config=conf/train_hkust_enocoder_conv1d_3_decoder_conv1d_1.yaml
  decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
  lm_config=conf/lm_default.yaml
  expname=${train_set}_${backend}_$(basename ${train_config%.*})
  expdir=$tgtdir/exp/${expname}
   ###add character lm
  if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
  fi
  recog_set="release"
  for rtask in $recog_set; do

   decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
   source-md/egs/asru2019/deal_with_recog_result_for_release.sh --steps 1-3 --bpe ${nbpe} --bpemodel ${bpemodel}.model  ${expdir}/${decode_dir} ${dict}
  done
fi

if [ ! -z $step60 ];then
    train_config=conf/train_hkust_enocoder_24_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step61 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_24_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev_new
# CH: %CER 11.16 [ 20185 / 180949, 510 ins, 901 del, 18774 sub ]
#  EN: %WER 33.70 [ 7554 / 22413, 380 ins, 430 del, 6744 sub ]
# MIX: %MER 13.64 [ 27739 / 203362, 890 ins, 1331 del, 25518 sub ]
# Result:11.16%,33.70%,13.64%
# dev
#  CH: %CER 6.60 [ 11435 / 173296, 426 ins, 634 del, 10375 sub ]
#  EN: %WER 19.80 [ 6013 / 30375, 371 ins, 326 del, 5316 sub ]
#  MIX: %MER 8.57 [ 17448 / 203671, 797 ins, 960 del, 15691 sub ]
# Result:6.60%,19.80%,8.57%
fi
if [ ! -z $step62 ];then
    train_config=conf/train_hkust_enocoder_36_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step63 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_36_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# single rtx2080 18109MiB
# dev_new
#  CH: %CER 11.97 [ 21651 / 180949, 574 ins, 1003 del, 20074 sub ]
#  EN: %WER 34.18 [ 7660 / 22413, 392 ins, 434 del, 6834 sub ]
# MIX: %MER 14.41 [ 29311 / 203362, 966 ins, 1437 del, 26908 sub ]
# Result:11.97%,34.18%,14.41%

# dev
# CH: %CER 7.08 [ 12272 / 173296, 446 ins, 731 del, 11095 sub ]
# EN: %WER 20.53 [ 6236 / 30375, 392 ins, 333 del, 5511 sub ]
#MIX: %MER 9.09 [ 18508 / 203671, 838 ins, 1064 del, 16606 sub ]
#Result:7.08%,20.53%,9.09%

fi

if [ ! -z $step64 ];then
    train_config=conf/train_hkust_enocoder_36_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step65 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_36_conv1d_3_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev_new
# CH: %CER 10.87 [ 19666 / 180949, 523 ins, 879 del, 18264 sub ]
# EN: %WER 33.07 [ 7411 / 22413, 390 ins, 426 del, 6595 sub ]
#MIX: %MER 13.31 [ 27077 / 203362, 913 ins, 1305 del, 24859 sub ]
#Result:10.87%,33.07%,13.31%
# dev
# CH: %CER 6.38 [ 11055 / 173296, 430 ins, 632 del, 9993 sub ]
# EN: %WER 19.21 [ 5836 / 30375, 346 ins, 337 del, 5153 sub ]
#MIX: %MER 8.29 [ 16891 / 203671, 776 ins, 969 del, 15146 sub ]
#Result:6.38%,19.21%,8.29%

fi

if [ ! -z $step66 ];then
    train_config=conf/train_hkust_enocoder_24_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step67 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_24_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev_new
# CH: %CER 10.85 [ 19627 / 180949, 537 ins, 915 del, 18175 sub ]
# EN: %WER 32.97 [ 7390 / 22413, 393 ins, 414 del, 6583 sub ]
# MIX: %MER 13.29 [ 27017 / 203362, 930 ins, 1329 del, 24758 sub ]
# Result:10.85%,32.97%,13.29%
# dev
# CH: %CER 6.47 [ 11204 / 173296, 408 ins, 674 del, 10122 sub ]
# EN: %WER 19.45 [ 5909 / 30375, 366 ins, 310 del, 5233 sub ]
# MIX: %MER 8.40 [ 17113 / 203671, 774 ins, 984 del, 15355 sub ]
# Result:6.47%,19.45%,8.40%
fi

if [ ! -z $step68 ];then
    train_config=conf/train_hkust_enocoder_28_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step69 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v3.sh ] && . ./path_v3.sh
    train_config=conf/train_hkust_enocoder_28_conv1d_5_decoder_conv1d_1.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev_new
# CH: %CER 10.97 [ 19845 / 180949, 544 ins, 873 del, 18428 sub ]
# EN: %WER 32.82 [ 7356 / 22413, 394 ins, 395 del, 6567 sub ]
#MIX: %MER 13.38 [ 27201 / 203362, 938 ins, 1268 del, 24995 sub ]
#Result:10.97%,32.82%,13.38%

# dev
# CH: %CER 6.48 [ 11232 / 173296, 432 ins, 661 del, 10139 sub ]
# EN: %WER 19.56 [ 5942 / 30375, 363 ins, 307 del, 5272 sub ]
#MIX: %MER 8.43 [ 17174 / 203671, 795 ins, 968 del, 15411 sub ]
#Result:6.48%,19.56%,8.43%
fi

if [ ! -z $step70 ];then
    train_config=conf/train_hkust_enocoder_36_conv1d_1_decoder_linear.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step71 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_36_conv1d_1_decoder_linear.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

# dev
# CH: %CER 6.39 [ 11072 / 173296, 450 ins, 625 del, 9997 sub ]
# EN: %WER 18.97 [ 5762 / 30375, 358 ins, 299 del, 5105 sub ]
# MIX: %MER 8.27 [ 16834 / 203671, 808 ins, 924 del, 15102 sub ]
# Result:6.39%,18.97%,8.27%
# dev_new
# CH: %CER 10.92 [ 19765 / 180949, 567 ins, 834 del, 18364 sub ]
# EN: %WER 33.00 [ 7396 / 22413, 414 ins, 395 del, 6587 sub ]
#MIX: %MER 13.36 [ 27161 / 203362, 981 ins, 1229 del, 24951 sub ]
#Result:10.92%,33.00%,13.36%
fi

if [ ! -z $step72 ];then
    train_config=conf/train_hkust_enocoder_36_conv1d_1_decoder_conv1d.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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
if [ ! -z $step73 ];then
    echo "LOG::: Decoding"
    [ -f ./path_v4.sh ] && . ./path_v4.sh
    train_config=conf/train_hkust_enocoder_36_conv1d_1_decoder_conv1d.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# dev_new
# CH: %CER 10.90 [ 19724 / 180949, 525 ins, 861 del, 18338 sub ]
# EN: %WER 32.96 [ 7388 / 22413, 429 ins, 355 del, 6604 sub ]
# MIX: %MER 13.33 [ 27112 / 203362, 954 ins, 1216 del, 24942 sub ]
#Result:10.90%,32.96%,13.33%


# dev
# CH: %CER 6.36 [ 11023 / 173296, 418 ins, 612 del, 9993 sub ]
# EN: %WER 19.17 [ 5824 / 30375, 379 ins, 293 del, 5152 sub ]
# MIX: %MER 8.27 [ 16847 / 203671, 797 ins, 905 del, 15145 sub ]
# Result:6.36%,19.17%,8.27%
fi

     
# because lm train text contain utt-id , it is not correct. so i will retrain a lm , in order to distinguish,  I call add _new suffix. 
###add character lm
if [ ! -z $step80 ]; then
   echo "stage 08: character LM Preparation"   
   lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}_new
   lmdict=${dict}
   mkdir -p ${lmdatadir}
    cut -f 2- -d" "  $tgtdir/data/train_trn/text | \
      spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " $tgtdir/data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/valid.txt
   echo "stage 08: character LM Preparation done"
fi


# use only 1 gpu
if [ ! -z $step81 ];then
    lm_config=conf/lm_default.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_new
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}

    lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}_new
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=0 \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${lm_ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/$lmexpdir \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --resume ${lm_resume} \
        --dict ${dict}
fi

if [ ! -z $step82 ];then
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_new
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    #[ -f ./path_v4.sh ] && . ./path_v4.sh
    #cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03,node06,node07,node08"
    cuda_cmd="slurm.pl --quiet --nodelist=node08"
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

if [ ! -z $step83 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    lm_config=conf/lm_default.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_new
    expdir=$tgtdir/exp/${expname}
    [ -d $expdir ] || mkdir -p ${expdir}
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}_new
    lmexpdir=${tgtdir}/exp/${lmexpname}
    [ -d $lmexpdir ] || mkdir -p ${lmexpdir}

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
    recog_set="dev_new dev"
    for rtask in $recog_set; do

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
            --rnnlm ${lmexpdir}/${lang_model}


        source-md/egs/asru2019/separated_score.sh --steps 1-4 --bpe ${nbpe} --bpemodel ${bpemodel}.model --wer true ${expdir}/${decode_dir} ${dict}


    pids+=($!) # store background pids
    done 
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done 
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
# dev_new

# CH: %CER 10.75 [ 19458 / 180949, 574 ins, 843 del, 18041 sub ]
#  EN: %WER 32.48 [ 7280 / 22413, 412 ins, 397 del, 6471 sub ]
# MIX: %MER 13.15 [ 26738 / 203362, 986 ins, 1240 del, 24512 sub ]
# Result:10.75%,32.48%,13.15%

# dev
#  CH: %CER 6.19 [ 10726 / 173296, 436 ins, 632 del, 9658 sub ]
#  EN: %WER 18.66 [ 5668 / 30375, 369 ins, 300 del, 4999 sub ]
# MIX: %MER 8.05 [ 16394 / 203671, 805 ins, 932 del, 14657 sub ]
# Result:6.19%,18.66%,8.05%
fi
