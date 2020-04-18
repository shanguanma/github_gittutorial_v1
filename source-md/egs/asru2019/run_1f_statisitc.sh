#!/bin/bash

##espnet version:0.5.3
#input feature:80-dim fbank(add pitch)
# model unit is sentencepiece(e.g.:bpe).
#            english unit:41 lowercase English letters word pieces
#            mandarin unit:3918 single Chinese characters.  
#            <unk> ' - . 

. path_v2.sh
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
tgtdir=run_1f_statisitc
train_set=train_trn_sp
train_dev=train_dev
recog_set="dev"
feat_tr_dir=${tgtdir}/${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${tgtdir}/${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

[ -d $tgtdir/data/cs200 ] || mkdir -p $tgtdir/data/cs200
[ -d $tgtdir/data/man500 ] || mkdir -p $tgtdir/data/man500
[ -d $tgtdir/data/librispeech ] || mkdir -p $tgtdir/data/librispeech
[ -d $tgtdir/data/dev ] || mkdir -p $tgtdir/data/dev

if [ ! -z $step01 ];then
    echo "stage 1: copy data"
    cp -r data/cs200/* $tgtdir/data/cs200
    cp -r data/dev/* $tgtdir/data/dev
    cp -r data/librispeech/* $tgtdir/data/librispeech
    cp -r data/man500/*  $tgtdir/data/man500 
fi
# speed-perturbed  cs200 set 
# make train validation set
if [ ! -z $step02 ]; then
     
     # speed-perturbed for train_trn set
     echo "LOG:::make speed-perturbed for train_trn set "
     utils/perturb_data_dir_speed.sh 0.9  ${tgtdir}/data/cs200  ${tgtdir}/data/temp1
     utils/perturb_data_dir_speed.sh 1.0  ${tgtdir}/data/cs200  ${tgtdir}/data/temp2
     utils/perturb_data_dir_speed.sh 1.1  ${tgtdir}/data/cs200  ${tgtdir}/data/temp3
     utils/combine_data.sh --extra-files utt2uniq ${tgtdir}/data/cs200_sp ${tgtdir}/data/temp1 ${tgtdir}/data/temp2 ${tgtdir}/data/temp3
     rm -r  ${tgtdir}/data/temp1  ${tgtdir}/data/temp2  ${tgtdir}/data/temp3
     echo "LOG:::make a train_dev and train_trn from cs200_sp set"
     # train_trn : train_dev =95:5
     utils/subset_data_dir_tr_cv.sh  --cv-spk-percent 5 $tgtdir/data/cs200_sp $tgtdir/data/cs200_sp_trn $tgtdir/data/${train_dev} || exit 1;

fi
# speed-perturbed for man500 set
if [ ! -z $step03 ];then
   echo "LOG:::make speed-perturbed for man500 set "
   utils/perturb_data_dir_speed.sh 0.9  ${tgtdir}/data/man500  ${tgtdir}/data/temp1
   utils/perturb_data_dir_speed.sh 1.0  ${tgtdir}/data/man500  ${tgtdir}/data/temp2
   utils/perturb_data_dir_speed.sh 1.1  ${tgtdir}/data/man500  ${tgtdir}/data/temp3
   utils/combine_data.sh --extra-files utt2uniq ${tgtdir}/data/man500_sp ${tgtdir}/data/temp1 ${tgtdir}/data/temp2 ${tgtdir}/data/temp3
   rm -r  ${tgtdir}/data/temp1  ${tgtdir}/data/temp2  ${tgtdir}/data/temp3
   utils/validate_data_dir.sh --no-feats $tgtdir/data/man500_sp || exit 1;
fi
# speed-perturbed for librispeech set
if [ ! -z $step04 ];then
   echo "LOG:::make speed-perturbed for librispeech set (it is 949.713 hours ) "
   utils/perturb_data_dir_speed.sh 0.9  ${tgtdir}/data/librispeech  ${tgtdir}/data/temp1
   utils/perturb_data_dir_speed.sh 1.0  ${tgtdir}/data/librispeech  ${tgtdir}/data/temp2
   utils/perturb_data_dir_speed.sh 1.1  ${tgtdir}/data/librispeech  ${tgtdir}/data/temp3
   utils/combine_data.sh --extra-files utt2uniq ${tgtdir}/data/librispeech_sp ${tgtdir}/data/temp1 ${tgtdir}/data/temp2 ${tgtdir}/data/temp3
   rm -r  ${tgtdir}/data/temp1  ${tgtdir}/data/temp2  ${tgtdir}/data/temp3
   utils/validate_data_dir.sh --no-feats $tgtdir/data/librispeech_sp || exit 1;

fi

# precess man500_sp, and get  include english words from man500 set.
if [ ! -z $step05 ];then
   # get include english words in main mandarian text(e.g.man500 set)  
   # awk '{sum +=$4-$3}END{print sum/3600}' $tgtdir/data/man500_sp_include_english/segments  :12.6894hours 
   source-md/egs/asru2019/re_eng_text.py $tgtdir/data/man500_sp/text > $tgtdir/data/man500_sp/english_text
   awk '{print $1 }' $tgtdir/data/man500_sp/english_text > $tgtdir/data/man500_sp/include_english_utt_list
   utils/subset_data_dir.sh --utt-list $tgtdir/data/man500_sp/include_english_utt_list $tgtdir/data/man500_sp $tgtdir/data/man500_sp_include_english
   # get rest utterance 
   awk '{print $1}' $tgtdir/data/man500_sp/utt2spk | utils/filter_scp.pl --exclude $tgtdir/data/man500_sp/include_english_utt_list > $tgtdir/data/man500_sp/man500_sp_rest_uttlist || exit 1;
   utils/subset_data_dir.sh --utt-list $tgtdir/data/man500_sp/man500_sp_rest_uttlist $tgtdir/data/man500_sp  $tgtdir/data/man500_sp_rest
   
fi

# get subset from man500_sp_rest and librispeech_sp, add to cs200_trn_sp as data augmentation.
if [ ! -z $step06 ];then
   # $tgtdir/data/man500_sp_325k : 
   # awk '{sum +=$4-$3}END{print sum/3600}' run_1d/data/man500_sp_325k/segments 
   # 291.1hours
   utils/subset_data_dir.sh $tgtdir/data/man500_sp_rest 325000  $tgtdir/data/man500_sp_325k
   #awk '{sum +=$4-$3}END{print sum/3600}' run_1d/data/librispeech_sp_90k/segments 
   # 283.159hours
   utils/subset_data_dir.sh $tgtdir/data/librispeech_sp 90000 $tgtdir/data/librispeech_sp_90k


fi

# Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features)
if [ ! -z $step07 ];then
    fbankdir=fbank
    for x in cs200_sp_trn; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50  \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done

fi

if [ ! -z $step08 ];then
    fbankdir=fbank
    for x in man500_sp_include_english; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 5  \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done

fi

if [ ! -z $step09 ];then
    fbankdir=fbank
    for x in man500_sp_325k; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50  \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done

fi


if [ ! -z $step10 ];then
    fbankdir=fbank
    for x in librispeech_sp_90k; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 50  \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
    done

fi


if [ ! -z $step11 ];then
  utils/combine_data.sh --extra-files utt2uniq $tgtdir/data/${train_set} ${tgtdir}/data/cs200_sp_trn  ${tgtdir}/data/man500_sp_include_english ${tgtdir}/data/man500_sp_325k $tgtdir/data/librispeech_sp_90k
  utils/validate_data_dir.sh --no-feats $tgtdir/data/${train_set}
fi
# make feature for recognition set
if [ ! -z $step12 ];then
   fbankdir=fbank
   for x in $recog_set; do
       steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20  \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
   done

fi

if [ ! -z $step13 ];then
   fbankdir=fbank
   for x in $train_dev; do
       steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 20  \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
   done


fi 
# make cmvn for $train_set
if [ ! -z $step14 ]; then
    #utils/spk2utt_to_utt2spk.pl data/train/spk2utt  > data/train/utt2spk
    # compute global CMVN for ${train_set} ,in order to dump . ${train_dev} and ${recog_set} are not required.
    echo "LOG:::compute global CMVN "
    compute-cmvn-stats scp:${tgtdir}/data/${train_set}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark
    # compute-cmvn-stats scp:${tgtdir}/data/${train_dev}_cut/feats.scp ${tgtdir}/data/${train_dev}_cut/cmvn.ark
    echo "LOG:::cmvn done"
fi

if [ ! -z $step15 ]; then
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

if [ ! -z $step16 ];then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "LOG:::stage 6: Dictionary  Preparation"
    mkdir -p ${tgtdir}/data/lang_char/

    echo "LOG:::in order to make a dictionary"
    echo "<unk> 1">${dict} #<unk> must be 1, 0 will be used for "blank" in CTC.
    # only cs200 text to get dict and lm, because its perplexity is very lower. 
    cat ${tgtdir}/data/cs200_sp_trn/text $tgtdir/data/man500_sp_include_english/text | cut -f 2- -d" " >${tgtdir}/data/lang_char/input.txt
    spm_train --input=${tgtdir}/data/lang_char/input.txt --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model --output_format=piece<${tgtdir}/data/lang_char/input.txt | tr ' ' '\n'| sort | uniq | awk '{print $0 " " NR+1}'>>${dict}
    wc -l ${dict}
    echo "LOG::: Dictionary  Preparation done"
fi
if [ ! -z $step17 ]; then
   echo " LOG::make json labels"
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
if [ ! -z $step18 ]; then
   echo "stage: character LM Preparation"   
   lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}
   lmdict=${dict}
   mkdir -p ${lmdatadir}
   cat ${tgtdir}/data/cs200_sp_trn/text $tgtdir/data/man500_sp_include_english/text | \
      spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " $tgtdir/data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/valid.txt
   echo "stage: character LM Preparation done"
fi
###add character lm
if [ ! -z $step19 ]; then
   echo "stage: character LM Preparation"   
   lmdatadir=$tgtdir/data/local/lm_train_${bpemodel}
   lmdict=${dict}
   mkdir -p ${lmdatadir}
   cat ${tgtdir}/data/cs200_sp_trn/text $tgtdir/data/man500_sp_include_english/text | \
      spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " $tgtdir/data/${train_dev}/text | spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/valid.txt
   echo "stage: character LM Preparation done"
fi
# use only 1 gpu
if [ ! -z $step20 ];then
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
    cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03" 
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
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
#conf/lm_default_v2.yaml
if [ ! -z $step21 ];then
    lm_config=conf/lm_default_v2.yaml
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
#conf/lm_default_v4.yaml
if [ ! -z $step22 ];then
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


if [ ! -z $step23 ];then
    train_config=conf/train_hkust_conv1d_statistic_large_v3.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
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


if [ ! -z $step24 ];then
   echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large_v3.yaml
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
# train_config=conf/train_hkust_conv1d_statistic_large_v3.yaml
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
# CH: %CER 7.93 [ 13748 / 173296, 665 ins, 1099 del, 11984 sub ]
# EN: %WER 23.51 [ 7142 / 30375, 453 ins, 461 del, 6228 sub ]
#MIX: %MER 10.26 [ 20890 / 203671, 1118 ins, 1560 del, 18212 sub ]
#Result:7.93%,23.51%,10.26%

fi




# because conf/train_hkust_conv1d_statistic_large_v3.yaml has overfit , so I want to regular method to reduece model complex.
# there are two common way.
# they are dropout and reduced number of encode layer
# first , I start to use dropout method. 
if [ ! -z $step25 ];then
    train_config=conf/train_hkust_conv1d_statistic_large_v4.yaml
    preprocess_config=conf/specaug.yaml
    decode_config=conf/decode_pytorch_transformer_csj_v2.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    echo "LOG::: Network Training"
    ngpu=1
    # use statistic 
    [ -f ./path_v1.sh ] && . ./path_v1.sh
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

if [ ! -z $step26 ];then
   echo "LOG::: Decoding"
    train_config=conf/train_hkust_conv1d_statistic_large_v4.yaml
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
# train_config=conf/train_hkust_conv1d_statistic_large_v4.yaml
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
#  cat log/run_1f_statisitc/steps26.log
# CH: %CER 89.08 [ 154367 / 173296, 2318 ins, 71631 del, 80418 sub ]
# EN: %WER 98.28 [ 29853 / 30375, 779 ins, 12868 del, 16206 sub ]
#MIX: %MER 90.45 [ 184220 / 203671, 3097 ins, 84499 del, 96624 sub ]
#Result:89.08%,98.28%,90.45% 
fi

