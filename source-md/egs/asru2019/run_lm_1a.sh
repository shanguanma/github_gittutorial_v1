#!/bin/bash


# this script is as baseline for asru2019 cs200 lm 
# it contains three kinds of lm from espnet official.
# They are default, seq_rnn, transformer
# in the epsnet 0.6.0 version, they are store at espnet/espnet/nets/pytorch_backend/lm

. path_v4.sh
. cmd.sh
steps=1
lm_ngpu=1
lmtag=
backend=pytorch
lm_resume=
tgtdir=run_lm_1a
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


#1. make train_dev from train
if [ ! -z $step01 ];then
     # make train_dev from train
     echo "LOG:::make a train_dev and train_trn from train set"
     mkdir -p $tgtdir/data/asru_cs200/train
     cp -r /home4/md510/w2019a/espnet-recipe/asru2019/data/cs200/*  $tgtdir/data/asru_cs200/train
     # Speakers, src=566, trn=538, cv=28
     utils/subset_data_dir_tr_cv.sh --cv-spk-percent 5 \
        $tgtdir/data/asru_cs200/train $tgtdir/data/asru_cs200/train_trn $tgtdir/data/asru_cs200/train_dev || exit 1;
fi

#2. train a bpe model
#3. get dictionary from train_trn
nbpe=3000
bpemode=bpe
train_set=train_trn
bpemodel=$tgtdir/data/asru_cs200/lang_char/${train_set}_${bpemode}${nbpe}
dict=$tgtdir/data/asru_cs200/lang_char/${train_set}_${nbpe}.units.txt
if [ ! -z $step02 ];then
    mkdir -p $tgtdir/data/asru_cs200/lang_char
    echo "<unk> 1">${dict} #<unk> must be 1, 0 will be used for "blank" in CTC.
    cut -f 2- -d" " $tgtdir/data/asru_cs200/train_trn/text>$tgtdir/data/asru_cs200/lang_char/input.txt
    spm_train --input=$tgtdir/data/asru_cs200/lang_char/input.txt \
              --vocab_size=${nbpe} --model_type=${bpemode} --model_prefix=${bpemodel} --input_sentence_size=100000000
    spm_encode --model=${bpemodel}.model \
               --output_format=piece<$tgtdir/data/asru_cs200/lang_char/input.txt | tr ' ' '\n'| sort | uniq | awk '{print $0 " " NR+1}'>>${dict}

fi

#4. enocding subword form 
if [ ! -z $step03 ];then
   echo "##LOG: character LM Preparation"   
   lmdatadir=$tgtdir/data/asru_cs200/lang_char
    cat $tgtdir/data/asru_cs200/lang_char/input.txt | \
      spm_encode --model=${bpemodel}.model --output_format=piece > ${lmdatadir}/train.txt
    cut -f 2- -d" " $tgtdir/data/asru_cs200/train_dev/text | spm_encode --model=${bpemodel}.model --output_format=piece >${lmdatadir}/valid.txt
   echo "##LOG: character LM Preparation done"
fi


# use only 1 gpu
# default style rnn lm
if [ ! -z $step04 ];then
    lm_config=conf/lm_default.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
    lmdatadir=$tgtdir/data/asru_cs200/lang_char
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
      CUDA_VISIBLE_DEVICES=1 \
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
# "perplexity": 7.795465139873305,
# "val_perplexity": 6.919177965031974
fi

# use only 1 gpu
# seq_rnn style rnn lm ,it is based on pytorch official example: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
if [ ! -z $step05 ];then
    lm_config=conf/lm_seq_rnn.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
    lmdatadir=$tgtdir/data/asru_cs200/lang_char
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
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
#  "perplexity": 20.593126528950883,
# "val_perplexity": 20.52139405223814

fi


# use only 1 gpu
# transformer style rnn lm
if [ ! -z $step06 ];then
    lm_config=conf/lm_transformer.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
    lmdatadir=$tgtdir/data/asru_cs200/lang_char
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=1 \
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

# "perplexity": 16.299547136035656,
# "val_perplexity": 14.844836985413613

fi

# use only 1 gpu
# seq_rnn style rnn lm ,it is based on pytorch official example: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
if [ ! -z $step07 ];then
    lm_config=conf/lm_seq_rnn_epochs40.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
    lmdatadir=$tgtdir/data/asru_cs200/lang_char
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=2 \
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
# "perplexity": 20.513057698212883,
# "val_perplexity": 20.30430326505143
fi

# use only 1 gpu
# transformer style  lm
if [ ! -z $step08 ];then
    lm_config=conf/lm_transformer_epochs40.yaml
    ###add character lm
    if [ -z ${lmtag} ]; then
    lmtag=$(basename ${lm_config%.*})
    fi
    lmexpname=train_rnnlm_${backend}_${lmtag}_${bpemode}${nbpe}
    lmexpdir=${tgtdir}/exp/${lmexpname}
    mkdir -p ${lmexpdir}
    cuda_cmd="slurm.pl --quiet --nodelist=node06"
    lmdatadir=$tgtdir/data/asru_cs200/lang_char
    echo " LOG:::training a lm model"
    # CUDA_VISIBLE_DEVICES=1 is specified gpu. its means I use gpu-id=1 gpu, gpu-id start from zeor.
    ${cuda_cmd} --gpu ${lm_ngpu} ${lmexpdir}/train.log \
    CUDA_VISIBLE_DEVICES=2 \
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

#  "perplexity": 14.324857221920158,
# "val_perplexity": 12.771112772785624

fi

