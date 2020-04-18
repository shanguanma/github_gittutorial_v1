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
resume=run_1h_statisitc/exp/train_trn_sp_pytorch_train_hkust_conv1d_statistic_large/results/snapshot.ep.2        # Resume the training from snapshot

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
tgtdir=run_1h_statisitc
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
   # copy dev data 
   cp -r /home4/md510/w2019a/espnet-recipe/asru2019/data/dev/*  ${tgtdir}/data/dev  
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
     # copy cs utterance from man500_sp set
     cp -r run_1f_statisitc/data/man500_sp_include_english ${tgtdir}/data/ 
     utils/combine_data.sh --extra-files utt2uniq ${tgtdir}/data/${train_set} ${tgtdir}/data/man500_sp_include_english ${tgtdir}/data/temp1 ${tgtdir}/data/temp2 ${tgtdir}/data/temp3
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
    cut -f 2- -d" " ${tgtdir}/data/train_trn/text ${tgtdir}/data/man500_sp_include_english/text>${tgtdir}/data/lang_char/input.txt
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
    cat $tgtdir/data/train_trn/text ${tgtdir}/data/man500_sp_include_english/text | \
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
    cuda_cmd="slurm.pl --quiet --nodelist=node08" 
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
#CH: %CER 6.54 [ 11340 / 173296, 468 ins, 630 del, 10242 sub ]
# EN: %WER 19.76 [ 6002 / 30375, 370 ins, 295 del, 5337 sub ]
#MIX: %MER 8.51 [ 17342 / 203671, 838 ins, 925 del, 15579 sub ]
#Result:6.54%,19.76%,8.51%


fi
if [ ! -z $step30 ];then
   [ -d $tgtdir/data/dev_new ] || mkdir -p $tgtdir/data/dev_new
   cp -r data/dev_new/* $tgtdir/data/dev_new/
fi

# make features for dev_new
if [ ! -z $step31 ];then
   fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
   for x in dev_new ; do
      steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done
fi

# dump features for dev_new
if [ ! -z $step32 ];then
   for rtask in dev_new; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 8 --do_delta $do_delta \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
   done
fi
# make josn format for dev_new
if [ ! -z $step33 ];then
    for rtask in dev_new; do
       feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
       data2json.sh --feat ${feat_recog_dir}/feats.scp \
           --bpecode ${bpemodel}.model ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data_${bpemode}${nbpe}.json

   done

fi
recog_set="dev_new"
if [ ! -z $step34 ];then
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


# dev_new
# CH: %CER 11.12 [ 20116 / 180949, 589 ins, 837 del, 18690 sub ]
# EN: %WER 34.36 [ 7701 / 22413, 441 ins, 344 del, 6916 sub ]
#MIX: %MER 13.68 [ 27817 / 203362, 1030 ins, 1181 del, 25606 sub ]
#Result:11.12%,34.36%,13.68%

fi

# inceasing decoder layer ,from 6 to 12.
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
# dev    9.29 
# dev CH 7.32
# dev EN 20.55


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
# dev     
# dev CH  
# dev EN  

fi

# test bigger encoder layer(e.g:48)















if [ ! -z $step150 ];then
    train_config=conf/train_hkust_conv1d_statistic_large_v1.yaml
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

if [ ! -z $step130 ];then
    echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large_v1.yaml
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
# train_config=conf/train_hkust_conv1d_statistic_large_v1.yaml
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
# dev CH
# dev EN

fi

