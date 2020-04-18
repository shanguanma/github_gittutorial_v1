#!/bin/bash

# using speed perturbed method

#input feature:80-dim fbank(add pitch)
# model unit is sentencepiece(e.g.:bpe).
#            english unit:41 lowercase English letters word pieces
#            mandarin unit:3918 single Chinese characters.  
#            <unk> ' - . 

. path_v7.sh
. cmd.sh
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

# general configuration
backend=pytorch
steps=
ngpu=1         # number of gpus ("0" uses cpu, otherwise use gpu)
debugmode=1
dumpdir=dump   # directory to dump full features
N=0            # number of minibatches to be used (mainly for debugging). "0" uses all minibatches.
verbose=0      # verbose option
resume=        # Resume the training from snapshot
seed=1

# feature configuration
do_delta=false

# sample filtering
min_io_delta=4  # samples with `len(input) - len(output) * min_io_ratio < min_io_delta` will be removed.

# rnnlm related
use_wordlm=true     # false means to train/use a character LM
lm_vocabsize=65000  # effective only for word LMs
lm_resume=          # specify a snapshot file to resume LM training
lmtag=              # tag for managing LMs

# decoding parameter
n_average=10 # use 1 for RNN models
recog_model=model.acc.best # set a model to be used for decoding: 'model.acc.best' or 'model.loss.best'

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

tgtdir=run_rnn_1a
train_set=train_si284
train_dev=test_dev93
train_test=test_eval92
recog_set="test_dev93 test_eval92"
feat_tr_dir=${tgtdir}/${dumpdir}/${train_set}/delta${do_delta}; mkdir -p ${feat_tr_dir}
feat_dt_dir=${tgtdir}/${dumpdir}/${train_dev}/delta${do_delta}; mkdir -p ${feat_dt_dir}

# data
wsj_corpus_root=/data/users/hhx502/wsj #  $wsj_corpos_root must contain a 'wsj0' and a 'wsj1' subdirectory for this to work.
# data prepared
# these files are kaldi format
# it is required by kaldi framework.


if [ ! -z $step01 ];then
   echo "stage 0: wsj Data preparation for kaldi"
   # wsj new format
   # $wsj_corpos_root must contain a 'wsj0' and a 'wsj1' subdirectory for this to work
   #1. prepare wsj data its output is data/local/data/ 
   local/cstr_wsj_data_prep.sh  $wsj_corpus_root
   
   # "nosp" refers to the dictionary before silence probabilities and pronunciation
   # probabilities are added. if you want to add silence probabilities and pronunciation
   # probabilities , you can see stage5 of https://github.com/kaldi-asr/kaldi/blob/master/egs/wsj/s5/run.sh
   #2. prepare dict ,this dict is cmu dict.
   local/wsj_prepare_dict.sh --dict-suffix "_nosp" || exit 1;
   #3. make lang for cmu dict 
   utils/prepare_lang.sh data/local/dict_nosp \
                         "<SPOKEN_NOISE>" data/local/lang_tmp_nosp data/lang_nosp || exit 1;
   #4 .make extend dict, its output is data/local/dict_nosp_larger 
   local/cstr_wsj_extend_dict.sh --dict-suffix "_nosp" /data/users/hhx502/wsj/wsj1/doc/
   #5. make lang for extend dict
   utils/prepare_lang.sh data/local/dict_nosp_larger \
                            "<SPOKEN_NOISE>" data/local/lang_tmp_nosp_larger data/lang_nosp_bd   
   #6. make lm, it is apra  format
   local/wsj_train_lms.sh --dict-suffix "_nosp"  || exit 1;
   #7. covert apra to G.fst for lm   
   local/wsj_format_local_lms.sh --lang-suffix "_nosp" || exit 1;
  
   #8. make data kaldi format, get wav.scp, utt2spk, spk2utt, text, spk2gender
   local/wsj_format_data.sh --lang-suffix "_nosp" || exit 1;  

fi

if [ ! -z $steo02 ];then
   echo "step02, wsj data preparation for espnet"
   # wsj new format
   # $wsj_corpos_root must contain a 'wsj0' and a 'wsj1' subdirectory for this to work
   #1. prepare wsj data its output is data/local/data
   local/cstr_wsj_data_prep.sh  $wsj_corpus_root
   #2.make data kaldi format, get wav.scp, utt2spk, spk2utt, text, spk2gender
   source-md/egs/wsj_e2e/wsj_format_data.sh  data/local/data  $tgtdir || exit 1; 
fi

# 2. make 83 dimension feature
if [ ! -z $step03 ];then
    ### Task dependent. You have to design training and dev sets by yourself.
    ### But you can utilize Kaldi recipes in most cases
    echo "stage 2: Feature Generation"
    fbankdir=fbank
    # Generate the fbank features;each frame of audio by a vector of 83 dimensions (80 Mel-filter bank coefficients 3 pitch features).
    for x in train_si284 test_dev93 test_eval92; do
        steps/make_fbank_pitch.sh --cmd "$train_cmd" --nj 10 --write_utt2num_frames true \
            ${tgtdir}/data/${x} ${tgtdir}/exp/make_fbank/${x} ${tgtdir}/${fbankdir}
        utils/fix_data_dir.sh ${tgtdir}/data/${x}
    done
fi
if [ ! -z $step04 ];then
   # compute global CMVN
    compute-cmvn-stats scp:${tgtdir}/data/${train_set}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark
    dump.sh --cmd "$train_cmd" --nj 32 --do_delta ${do_delta} \
        ${tgtdir}/data/${train_set}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/train ${feat_tr_dir}
    dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
        ${tgtdir}/data/${train_dev}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/dev ${feat_dt_dir}
    for rtask in ${recog_set}; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}; mkdir -p ${feat_recog_dir}
        dump.sh --cmd "$train_cmd" --nj 4 --do_delta ${do_delta} \
            ${tgtdir}/data/${rtask}/feats.scp ${tgtdir}/data/${train_set}/cmvn.ark ${tgtdir}/exp/dump_feats/recog/${rtask} \
            ${feat_recog_dir}
    done
fi

dict=${tgtdir}/data/lang_1char/${train_set}_units.txt
nlsyms=${tgtdir}/data/lang_1char/non_lang_syms.txt

echo "dictionary: ${dict}"
if [ ! -z $step05 ]; then
    ### Task dependent. You have to check non-linguistic symbols used in the corpus.
    echo "stage 2: Dictionary and Json Data Preparation"
    mkdir -p ${tgtdir}/data/lang_1char/

    echo "make a non-linguistic symbol list"
    cut -f 2- ${tgtdir}/data/${train_set}/text | tr " " "\n" | sort | uniq | grep "<" > ${nlsyms}
    cat ${nlsyms}

    echo "make a dictionary"
    echo "<unk> 1" > ${dict} # <unk> must be 1, 0 will be used for "blank" in CTC
    text2token.py -s 1 -n 1 -l ${nlsyms} ${tgtdir}/data/${train_set}/text | cut -f 2- -d" " | tr " " "\n" \
    | sort | uniq | grep -v -e '^\s*$' | awk '{print $0 " " NR+1}' >> ${dict}
    wc -l ${dict}

    echo "make json files"
    data2json.sh --feat ${feat_tr_dir}/feats.scp --nlsyms ${nlsyms} \
         ${tgtdir}/data/${train_set} ${dict} > ${feat_tr_dir}/data.json
    data2json.sh --feat ${feat_dt_dir}/feats.scp --nlsyms ${nlsyms} \
         ${tgtdir}/data/${train_dev} ${dict} > ${feat_dt_dir}/data.json
    for rtask in ${recog_set}; do
        feat_recog_dir=${tgtdir}/${dumpdir}/${rtask}/delta${do_delta}
        data2json.sh --feat ${feat_recog_dir}/feats.scp \
            --nlsyms ${nlsyms} ${tgtdir}/data/${rtask} ${dict} > ${feat_recog_dir}/data.json
    done
fi
if [ ! -z $step06 ];then
     ### Filter out short samples which lead to `loss_ctc=inf` during training
    ###  with the specified configuration.
    # Samples satisfying `len(input) - len(output) * min_io_ratio < min_io_delta` will be pruned.
    train_config=conf/wsj/train_rnn.yaml
    preprocess_config=conf/wsj/no_preprocess.yaml ## use conf/specaug.yaml for data augmentation
    source-md/egs/wsj_e2e/filtering_samples.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --data-json ${feat_tr_dir}/data.json \
        --mode-subsample "asr" \
        --arch-subsample "rnn" \
        ${min_io_delta:+--min-io-delta $min_io_delta} \
        --output-json-path ${feat_tr_dir}/data.json
fi

# It takes about one day. If you just want to do end-to-end ASR without LM,
# you can skip this and remove --rnnlm option in the recognition (stage 5)

if [ ! -z $step07 ]; then
    echo "stage : LM Preparation"

    if [ ${use_wordlm} = true ]; then
        lmdatadir=$tgtdir/data/local/wordlm_train
        lmdict=${lmdatadir}/wordlist_${lm_vocabsize}.txt
        mkdir -p ${lmdatadir}
        cut -f 2- -d" " $tgtdir/data/${train_set}/text > ${lmdatadir}/train_trans.txt
        zcat /data/users/hhx502/wsj/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
                | grep -v "<" | tr "[:lower:]" "[:upper:]" > ${lmdatadir}/train_others.txt
        cut -f 2- -d" " $tgtdir/data/${train_dev}/text > ${lmdatadir}/valid.txt
        cut -f 2- -d" " $tgtdir/data/${train_test}/text > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
        text2vocabulary.py -s ${lm_vocabsize} -o ${lmdict} ${lmdatadir}/train.txt
    else
        lmdatadir=$tgtdir/data/local/lm_train
        lmdict=${dict}
        mkdir -p ${lmdatadir}
        text2token.py -s 1 -n 1 -l ${nlsyms} $tgtdir/data/${train_set}/text \
            | cut -f 2- -d" " > ${lmdatadir}/train_trans.txt
        zcat /data/users/hhx502/wsj/wsj1/doc/lng_modl/lm_train/np_data/{87,88,89}/*.z \
            | grep -v "<" | tr "[:lower:]" "[:upper:]" \
            | text2token.py -n 1 | cut -f 2- -d" " > ${lmdatadir}/train_others.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} $tgtdir/data/${train_dev}/text \
            | cut -f 2- -d" " > ${lmdatadir}/valid.txt
        text2token.py -s 1 -n 1 -l ${nlsyms} $tgtdir/data/${train_test}/text \
                | cut -f 2- -d" " > ${lmdatadir}/test.txt
        cat ${lmdatadir}/train_trans.txt ${lmdatadir}/train_others.txt > ${lmdatadir}/train.txt
    fi
fi
# It takes about one day. If you just want to do end-to-end ASR without LM,
if [ ! -z $step08 ];then
    lm_config=conf/wsj/lm.yaml
    if [ -z ${lmtag} ]; then
     lmtag=$(basename ${lm_config%.*})
     if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
     fi
   fi
   lmexpname=train_rnnlm_${backend}_${lmtag}
   lmexpdir=$tgtdir/exp/${lmexpname}
   mkdir -p ${lmexpdir}
    ${cuda_cmd} --gpu ${ngpu} ${lmexpdir}/train.log \
        lm_train.py \
        --config ${lm_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --verbose 1 \
        --outdir ${lmexpdir} \
        --tensorboard-dir tensorboard/${lmexpname} \
        --train-label ${lmdatadir}/train.txt \
        --valid-label ${lmdatadir}/valid.txt \
        --test-label ${lmdatadir}/test.txt \
        --resume ${lm_resume} \
        --dict ${lmdict}
fi


if  [ ! -z $step09 ]; then
    echo "stage : Network Training"
    # config files
    preprocess_config=conf/wsj/no_preprocess.yaml  #conf/specaug.yaml #for data augmentation
    train_config=conf/wsj/train_rnn.yaml
    lm_config=conf/wsj/lm.yaml
    decode_config=conf/wsj/decode_rnn.yaml
    cuda_cmd="slurm.pl --quiet --nodelist=node05"
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}
    ${cuda_cmd} --gpu ${ngpu} ${expdir}/train.log \
        asr_train.py \
        --config ${train_config} \
        --preprocess-conf ${preprocess_config} \
        --ngpu ${ngpu} \
        --backend ${backend} \
        --outdir ${expdir}/results \
        --tensorboard-dir tensorboard/${expname} \
        --debugmode ${debugmode} \
        --dict ${dict} \
        --debugdir ${expdir} \
        --minibatches ${N} \
        --verbose ${verbose} \
        --resume ${resume} \
        --seed ${seed} \
        --train-json ${feat_tr_dir}/data.json \
        --valid-json ${feat_dt_dir}/data.json
fi
if [ ! -z $step10 ]; then
    echo "stage : Decoding"
    # config files
    preprocess_config=conf/wsj/no_preprocess.yaml  # use conf/specaug.yaml for data augmentation
    train_config=conf/wsj/train_rnn.yaml
    lm_config=conf/wsj/lm.yaml
    decode_config=conf/wsj/decode_rnn.yaml
    expname=${train_set}_${backend}_$(basename ${train_config%.*})_$(basename ${preprocess_config%.*})
    if ${do_delta}; then
        expname=${expname}_delta
    fi
    expdir=$tgtdir/exp/${expname}
    mkdir -p ${expdir}

    if [ -z ${lmtag} ]; then
     lmtag=$(basename ${lm_config%.*})
     if [ ${use_wordlm} = true ]; then
        lmtag=${lmtag}_word${lm_vocabsize}
     fi
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}
    lmexpdir=$tgtdir/exp/${lmexpname}
    mkdir -p ${lmexpdir}
    nj=32
    if [[ $(get_yaml.py ${train_config} model-module) = *transformer* ]]; then
        recog_model=model.last${n_average}.avg.best
        average_checkpoints.py --backend ${backend} \
                               --snapshots ${expdir}/results/snapshot.ep.* \
                               --out ${expdir}/results/${recog_model} \
                               --num ${n_average}
    fi

    pids=() # initialize pids
    for rtask in ${recog_set}; do
    (
        decode_dir=decode_${rtask}_$(basename ${decode_config%.*})_${lmtag}
        if [ ${use_wordlm} = true ]; then
            recog_opts="--word-rnnlm ${lmexpdir}/rnnlm.model.best"
        else
            recog_opts="--rnnlm ${lmexpdir}/rnnlm.model.best"
        fi
        feat_recog_dir=$tgtdir/${dumpdir}/${rtask}/delta${do_delta}

        # split data
        splitjson.py --parts ${nj} ${feat_recog_dir}/data.json

        #### use CPU for decoding
        ngpu=0

        ${decode_cmd} JOB=1:${nj} ${expdir}/${decode_dir}/log/decode.JOB.log \
            asr_recog.py \
            --config ${decode_config} \
            --ngpu ${ngpu} \
            --backend ${backend} \
            --recog-json ${feat_recog_dir}/split${nj}utt/data.JOB.json \
            --result-label ${expdir}/${decode_dir}/data.JOB.json \
            --model ${expdir}/results/${recog_model}  \
            ${recog_opts}

        score_sclite.sh --wer true --nlsyms ${nlsyms} ${expdir}/${decode_dir} ${dict}

    ) &
    pids+=($!) # store background pids
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((++i)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false
    echo "Finished"
# WER 
#  cat log/run_rnn_1a/step10.log 
# model unit is letter and using word rnnlm. 
# test_eavl192 : 5.1
# test_dev93 : 8.7
fi
