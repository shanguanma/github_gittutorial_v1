#!/bin/bash

# Ma Duo 2020-3-13

# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
#set -u
set -o pipefail

log() {
    local fname=${BASH_SOURCE[1]##*/}
    echo -e "$(date '+%Y-%m-%dT%H:%M:%S') (${fname}:${BASH_LINENO[0]}:${FUNCNAME[1]}) $*"
}

min() {
  local a b
  a=$1
  for b in "$@"; do
      if [ "${b}" -le "${a}" ]; then
          a="${b}"
      fi
  done
  echo "${a}"
}

SECONDS=0

cmd="slurm.pl  --quiet --exclude=node06,node05"
#steps=1
stage=1
stop_stage=120
chain_stage=0
chain_model_train_stage=-10
num_epochs=20
decode_nj=200
nj=20

# releated  ngram 
n_order=4          # order of  n-gram lm, for example 4-gram ,its order is 4, default is 4,  you usually can set 3 or 4.
oov_symbol="<UNK>" #  oov symbol for making  maxent lm.

# releated dict
use_pp=true             # we compute the pronunciation and silence probabilities from training data,
                        # and re-create the lang and lang_test directory.
# realted gmm 
shortest_utt_num=20000  # these shortest utterances is used to do train mono, it is useful for alignment

wave_sample=16k     # wave sample frequency, it may be 8k or 16k
# releated chain model
feats_type=perturb  # using speed perturb and volumn perturb for data augmentation
                    # default is perturb

frontend_type=nocodec # codec , nocodec 

nnet3_affix=""      # i-vector folder affix
sp_suffix=_sp       # # when using speed perturb and volumn perturb. it should be "_sp" ,else "_nosp" 
pp_suffix=_pp       # when using pp dictionary , it should be "_pp" ,else ""
dnn_size=small      # if train data less 50  hours ,it will be "small", else , it will "normal".
hmm_gmm_version=v1

# [Task dependent] Set the datadir name created by
src_train_data_dir=
src_test_data_dir=
train_set=                  # train set name
test_sets=
srcdictdir=
lm_train_text=              # Use  train set text to do  lm training if not specified.
tgtdir=                     #  root folder of the entire asr system 
suffix=""                   # you can set it, then run different model in same tgtdir,

help_message=$(cat << EOF
Usage: $0 --src_train_data_dir <train data path> --src_test_data_dir <test data path> --train-set <train_set_name> --test_sets <test_set_names> --srcdictdir <dictdir path > --tgtdir <tgtdir_name> 

Options:
    # General configuration
     --cmd                       # preocess environmet (default="${cmd}")
     --stage                     # Processes starts from the specified stage (default="${stage}").
     --stop_stage                # Processes is stopped at the specified stage (default="${stop_stage}").
     --chain_stage=0             # chain model script of nnnet3 processes start  from the specified stage (default=0)
     --chain_model_train_stage   # chain model training iter from the specified stage (default=-10) 
     --nj         # The number of parallel jobs (default="${nj}"). 
    #  ngram releated 
    --n_order     #  order of  n-gram lm, for example 4-gram ,its order is 4, (default=$n_order),  you usually can set 3 or 4.
    # releated dict
    --use_pp            # whether we compute the pronunciation and silence probabilities from training data, default is true, (default="${use_pp}")
                        # and re-create the lang and lang_test directory.
    # realted gmm 
    --shortest_utt_num  # these utterances is used to do train mono, it is useful for alignment. this value
                        # it is usually about 1/5 the entire train set text utterance. type (800,default="$shortest_utt_num")
    

    --wave_sample     # wave sample frequency, type ( 8k or 16k, default=${wave_sample})
    # releated chain model
    --feats_type  #whether using speed perturb and volumn perturb for data augmentation # (default=$feats_type )
    --nnet3_affix      # i-vector folder affix (default="$nnet3_affix")
    --sp_suffix        # # when using speed perturb and volumn perturb. it should be "_sp" ,else "_nosp" (default="$sp_suffix")
    --small_dnn        # if train data less 50  hours ,it will be true, else , it will false. (default=${small_dnn})

    # [Task dependent] Set the datadir name created
    --src_train_data_dir  # path of train kaldi format folder (required).
    --src_test_data_dir   # path of test kaldi format folder (required). 
    --train_set     # Name of training set (required)..
    --test_sets     # Names of evaluation sets (required).
    --srcdictdir    # pronunciation dictionary (required).
    --lm_train_text # text file for training lm, default it is empty, it will only use train text to train lm (default="${lm_train_text}") 
    --tgtdir        # the entire asr system folder (required).
    --suffix        # the suffix of exp folder and data folder. (default="${suffix}").
    
      
 
EOF
)

log "$0 $*"
. utils/parse_options.sh

if [ $# -ne 0 ]; then
    log "${help_message}"
    
    log "Error: No positional arguments are required. all path varibale must be add """
    exit 2
fi

#steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
#  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
#        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
#      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }} print $steps;' 2>/dev/null)  || exit 1
#
#if [ ! -z "$steps" ]; then
#  for x in $(echo $steps|sed 's/[,:]/ /g'); do
#    index=$(printf "%02d" $x);
#    declare step$index=1
#  done
#fi

. path.sh
. cmd.sh

# Check required arguments
[ -z "${src_train_data_dir}" ] &&  { log "${help_message}"; log "Error: --src_train_data_dir is required" ; exit 2; };
[ -z "${src_test_data_dir}" ] &&  { log "${help_message}"; log "Error: --src_test_data_dir is required" ; exit 2; };
[ -z "${train_set}" ] && { log "${help_message}"; log "Error: --train_set is required"; exit 2; };
[ -z "${test_sets}" ] && { log "${help_message}"; log "Error: --test_sets is required"; exit 2; };
[ -z "${srcdictdir}" ] &&  { log "${help_message}"; log "Error: --srcdictdir is required" ; exit 2; };
[ -z "${tgtdir}" ] &&  { log "${help_message}"; log "Error: --tgtdir  is required" ; exit 2; };

exp_root=$tgtdir/exp${suffix}
lang=$tgtdir/data$suffix/lang
#dictdir=$tgtdir/data$suffix/maison2_dict
dictdir=$tgtdir/data$suffix/dict

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ];then
    log "Stage 2: train Data preparation"
    for part in ${train_set}; do
        utils/copy_data_dir.sh ${src_train_data_dir}/${part} $tgtdir/data$suffix/${part}|| exit 1;
        utils/validate_data_dir.sh --no-feats $tgtdir/data$suffix/${part} || exit 1;
    done
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ];then
    log "Stage 3: test Data preparation"
    for part in  ${test_sets}; do
        utils/copy_data_dir.sh ${src_test_data_dir}/${part} $tgtdir/data$suffix/${part}_test|| exit 1;
        utils/validate_data_dir.sh --no-feats $tgtdir/data$suffix/${part}_test || exit 1;
    done

fi 

# make lang
if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 4 ];then
  log "Stage 1: lang prepareation"
  [ -d $dictdir ] || mkdir -p $dictdir
  [ -d $exp_root ] || mkdir -p $exp_root
  # below command is very danger, You must guarantee that this path is absolutely correct. otherwise, it will copy /* $dictdir , note / is root folder. 
  cp -r $srcdictdir/*  $dictdir/
  utils/validate_dict_dir.pl ${dictdir} || { echo "## ERROR : failed to validating dict '$dictdir'" && exit 1;  }
  utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ];then
 
  if [ "${wave_sample}" = 8k ]; then
   log "Stage 4: 13 dimension mfcc feature preparation for 8k train data"
    for part in ${train_set}; do
       _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${part}/utt2spk" wc -l)")
        steps/make_mfcc.sh  --nj "${_nj}" --cmd "${train_cmd}" \
                                 --mfcc-config  conf/mfcc_8k.conf  "$tgtdir/data$suffix/${part}"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}"
        steps/compute_cmvn_stats.sh  "$tgtdir/data$suffix/${part}"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}"
   done 
  elif [ "${wave_sample}" = 16k ];then
   log "Stage 4: 13 dimension mfcc feature preparation for 16k train data"
    for part in ${train_set}; do
       _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${part}/utt2spk" wc -l)")
        steps/make_mfcc.sh  --nj "${_nj}" --cmd "${train_cmd}" \
                                 --mfcc-config  conf/mfcc.conf  "$tgtdir/data$suffix/${part}"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}"
        steps/compute_cmvn_stats.sh "$tgtdir/data$suffix/${part}"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}"
   done

  else
    log "Error: not supported: --wave-sample ${wave_sample}"
    exit 2
 fi
fi


if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ];then

  if [ "${wave_sample}" = 8k ]; then
   log "Stage 5: 13 dimension mfcc feature preparation for 8k test data"
    for part in  ${test_sets}; do
       _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${part}_test/utt2spk" wc -l)")
        steps/make_mfcc.sh  --nj "${_nj}" --cmd "${train_cmd}" \
                                 --mfcc-config  conf/mfcc_8k.conf  "$tgtdir/data$suffix/${part}_test"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}_test"
        steps/compute_cmvn_stats.sh  "$tgtdir/data$suffix/${part}_test"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}_test"
   done
  elif [ "${wave_sample}" = 16k ];then
   log "Stage 5: 13 dimension mfcc feature preparation for 16k test data"
    for part in ${test_sets}; do
       _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${part}_test/utt2spk" wc -l)")
        steps/make_mfcc.sh  --nj "${_nj}" --cmd "${train_cmd}" \
                                 --mfcc-config  conf/mfcc.conf  "$tgtdir/data$suffix/${part}_test"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}_test"
        steps/compute_cmvn_stats.sh "$tgtdir/data$suffix/${part}_test"
        utils/fix_data_dir.sh "$tgtdir/data$suffix/${part}_test"
   done

  else
    log "Error: not supported: --wave-sample ${wave_sample}"
    exit 2
 fi
fi

# Use the same text as ASR for lm training if not specified.
[ -z "${lm_train_text}" ] && lm_train_text="$tgtdir/data$suffix/${train_set}/text"

###############################################################################
# pepared new lang_test 
# more detail ,you can read /home4/md510/w2019a/kaldi-recipe/lm/data/lm/perplexities.txt
#                           /home4/md510/w2019a/kaldi-recipe/lm/data/lm/ 
#                           /home4/md510/w2019a/kaldi-recipe/lm/train_lms_srilm.sh 
###############################################################################

lmdir=$tgtdir/data$suffix/local/lm
train_data=${datadir}/$train_set
# prepared G.fst
[ -d $lmdir ] || mkdir -p $lmdir
words_file=$lang/words.txt
if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
   log "Stage 6: prepared train.txt and vocab for train maxent lm "
   log "Using words file: $words_file"
   sort $words_file | awk '{print $1}' | grep -v '\#0' | grep -v '<eps>' | grep -v -F "$oov_symbol" > $lmdir/vocab
   cat ${lm_train_text} | cut -f2- -d' '> $lmdir/train.txt
fi
if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  log "-------------------"
  log "Stage 7: make Maxent ${n_order}ggrams"
  log "-------------------"
  # sed 's/'${oov_symbol}'/<unk>/g' means: using <unk> to replace ${oov_symbol}
  sed 's/'${oov_symbol}'/<unk>/g' $lmdir/train.txt | \
    ngram-count -lm - -order ${n_order} -text - -vocab $lmdir/vocab -unk -sort -maxent -maxent-convert-to-arpa| \
   sed 's/<unk>/'${oov_symbol}'/g' | gzip -c > $lmdir/${n_order}gram.me.gz 
  log "## LOG : done with '$lmdir/${n_order}gram.me.gz'"
fi


lang_test=$tgtdir/data$suffix/local/lang_test
[ -d $lang_test ] || mkdir -p $lang_test
if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  log "Stage 8: make G.fst for lang_test"
  utils/format_lm.sh $tgtdir/data$suffix/lang $tgtdir/data$suffix/local/lm/${n_order}gram.me.gz \
    $dictdir/lexiconp.txt $lang_test
  utils/validate_lang.pl $lang_test
  log "## LOG : done with '$lang_test'"
fi

# select shortest utt to do mono, it is useful.
if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ];then
  log " Step 9 :select ${shortest_utt_num} shortest utterances to train mono gmm"
  utils/subset_data_dir.sh  \
      --shortest  $tgtdir/data$suffix/${train_set} ${shortest_utt_num} \
        $tgtdir/data$suffix/${train_set}_${shortest_utt_num}short
fi


if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ];then
  log " Stage 10: train mono...."
  steps/train_mono.sh  --boost-silence 1.25 --nj  10  --cmd "$cmd" \
    $tgtdir/data$suffix/${train_set}_${shortest_utt_num}short  $lang $exp_root/mono

fi

if [ ${stage} -le 11 ] && [ ${stop_stage} -ge 11 ]; then
  log "Stage 11: ali mono and train deltas ...."
  steps/align_si.sh --boost-silence 1.25  --nj 10 --cmd "$cmd" \
    $tgtdir/data$suffix/${train_set}_${shortest_utt_num}short $lang $exp_root/mono $exp_root/mono_ali

  steps/train_deltas.sh --cmd "$cmd" --boost-silence 1.25 \
    2000 10000  $tgtdir/data$suffix/${train_set}  $lang $exp_root/mono_ali $exp_root/tri1

fi

if [ ${stage} -le 12 ] && [ ${stop_stage} -ge 12 ]; then
   if [ "${hmm_gmm_version}" = "v1" ];then
     log "From here: using hmm-gmm version is v1........ at tri2 stage, it use lda_mllt without splice frames  "
     log "Stage 12: tri1 ali and lda_mllt training ....."
     steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                    $tgtdir/data$suffix/${train_set}  $lang $exp_root/tri1 $exp_root/tri1_ali

     steps/train_lda_mllt.sh --cmd "$cmd" \
      2500 15000 $tgtdir/data$suffix/${train_set} $lang $exp_root/tri1_ali $exp_root/tri2
   elif [ "${hmm_gmm_version}" = "v2" ];then
     log "From here: using hmm-gmm version is v1........ at tri2 stage, it use delta+delta-delta feature to train a tri2 system"
     log "Stage 12: tri1 ali and delta+delta-delta  training ....."
     steps/align_si.sh --nj $nj --cmd "$train_cmd" \
                    $tgtdir/data$suffix/${train_set}  $lang $exp_root/tri1 $exp_root/tri1_ali
     steps/train_deltas.sh --cmd "$cmd" \
      2500 15000 $tgtdir/data$suffix/${train_set} $lang $exp_root/tri1_ali $exp_root/tri2
   else
     log "Error: not supported: --hmm_gmm_version ${hmm_gmm_version}"
     exit 2;
   fi
 fi

if [ ${stage} -le 13 ] && [ ${stop_stage} -ge 13 ]; then
  log "Stage 13: tri2 ali and lda_mllt training ....."
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    $tgtdir/data$suffix/${train_set} $lang $exp_root/tri2 $exp_root/tri2_ali

  steps/train_lda_mllt.sh --cmd "$cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     2500 15000 $tgtdir/data$suffix/${train_set} $lang $exp_root/tri2_ali $exp_root/tri3

 fi

 if [ ${stage} -le 14 ] && [ ${stop_stage} -ge 14 ]; then
  log "Stage 14: using fmllr for tri3 ali  and train sat ...."
  steps/align_fmllr.sh --nj 40 --cmd "$cmd" \
    $tgtdir/data$suffix/${train_set}  $lang $exp_root/tri3 $exp_root/tri3_ali || exit 1;

  steps/train_sat.sh --cmd "$cmd" \
    2500 15000 $tgtdir/data$suffix/${train_set}  $lang $exp_root/tri3_ali $exp_root/tri4 || exit 1;

 fi

 if "${use_pp}"; then
   if [ ${stage} -le 15 ] && [ ${stop_stage} -ge 15 ]; then
     log "Stage 15:  Now we compute the pronunciation and silence probabilities from training data"
     steps/get_prons.sh --cmd "$cmd" \
      $tgtdir/data$suffix/${train_set}  $lang $exp_root/tri4

     utils/dict_dir_add_pronprobs.sh --max-normalize true \
       $dictdir \
       $exp_root/tri4/pron_counts_nowb.txt $exp_root/tri4/sil_counts_nowb.txt \
       $exp_root/tri4/pron_bigram_counts_nowb.txt $tgtdir/data$suffix/local/dict_pp

     log "Stage 15 : and re-create the lang directory"
     utils/prepare_lang.sh $tgtdir/data$suffix/local/dict_pp \
      "<unk>" $tgtdir/data$suffix/lang_pp/tmp $tgtdir/data$suffix/lang_pp

     log "Stage 15 : and re-create the lang_test directory"
     utils/format_lm.sh $tgtdir/data$suffix/lang_pp $tgtdir/data$suffix/local/lm/${n_order}gram.me.gz \
       $dictdir/lexiconp.txt $tgtdir/data$suffix/lang_test_pp

   fi
 fi



# make i-vector
# 1. get lower train data, in order to get ali, it is used as input data of steps/align_fmllr_lats.sh  
# it is 13 dim mfcc 
if [ ${stage} -le 16 ] && [ ${stop_stage} -ge 16 ]; then
   if [ "${feats_type}" = perturb ]; then
      if [ "${frontend_type}" = codec ];then
        if [ "${wave_sample}" = 8k ]; then
          log "Stage 16: using speed perturb for 8k codec train data augmentation,it is 13 dim mfcc "
           for datadir in ${train_set}; do
             /home4/md510/package/source-md/asr_frontend/data_level/codec/perturb_data_dir_speed_3way.sh \
                  $tgtdir/data$suffix/${datadir} $tgtdir/data$suffix/${datadir}_sp
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp

             _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${datadir}_sp/utt2spk" wc -l)")
             steps/make_mfcc.sh --cmd "$cmd" --nj ${_nj} --mfcc-config conf/mfcc_8k.conf \
               $tgtdir/data$suffix/${datadir}_sp || exit 1;
             steps/compute_cmvn_stats.sh \
               $tgtdir/data$suffix/${datadir}_sp || exit 1;
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp
           done
         elif [ "${wave_sample}" = 16k ];then
           log "Stage 16: using speed perturb for 16k train data augmentation, it is 13 dim mfcc "
           for datadir in ${train_set}; do
             /home4/md510/package/source-md/asr_frontend/data_level/codec/perturb_data_dir_speed_3way.sh \
                  $tgtdir/data$suffix/${datadir} $tgtdir/data$suffix/${datadir}_sp
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp

             _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${datadir}_sp/utt2spk" wc -l)")
             steps/make_mfcc.sh --cmd "$cmd" --nj ${_nj} --mfcc-config conf/mfcc.conf \
               $tgtdir/data$suffix/${datadir}_sp  || exit 1;
             steps/compute_cmvn_stats.sh \
               $tgtdir/data$suffix/${datadir}_sp  || exit 1;
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp
           done
         else
           log "Error: not supported: --wave-sample ${wave_sample}"
           exit 2
         fi
      elif [ "${frontend_type}" = nocodec ];then
         if [ "${wave_sample}" = 8k ]; then
           log "Stage 16: using speed perturb for 8k train data augmentation,it is 13 dim mfcc "
           for datadir in ${train_set}; do
             utils/data/perturb_data_dir_speed_3way.sh $tgtdir/data$suffix/${datadir} $tgtdir/data$suffix/${datadir}_sp
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp

             _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${datadir}_sp/utt2spk" wc -l)")
             steps/make_mfcc.sh --cmd "$cmd" --nj ${_nj} --mfcc-config conf/mfcc_8k.conf \
               $tgtdir/data$suffix/${datadir}_sp || exit 1;
             steps/compute_cmvn_stats.sh \
               $tgtdir/data$suffix/${datadir}_sp || exit 1;
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp
           done
         elif [ "${wave_sample}" = 16k ];then
           log "Stage 16: using speed perturb for 16k train data augmentation, it is 13 dim mfcc "
           for datadir in ${train_set}; do
             utils/data/perturb_data_dir_speed_3way.sh $tgtdir/data$suffix/${datadir} $tgtdir/data$suffix/${datadir}_sp
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp

             _nj=$(min "${nj}" "$(<"$tgtdir/data$suffix/${datadir}_sp/utt2spk" wc -l)")
             steps/make_mfcc.sh --cmd "$cmd" --nj ${_nj} --mfcc-config conf/mfcc.conf \
               $tgtdir/data$suffix/${datadir}_sp  || exit 1;
             steps/compute_cmvn_stats.sh \
               $tgtdir/data$suffix/${datadir}_sp  || exit 1;
             utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_sp
           done
         else
           log "Error: not supported: --wave-sample ${wave_sample}"
           exit 2
         fi
      else
        log "Error: not supported: --frontend-type ${frontend_type}"
        exit 2 
        fi 
   else 
      if [ "${wave_sample}" = 8k ]; then
        log "Stage 16: We don't use speed perturb for 8k train data, it is 13 dim mfcc  "
        for datadir in ${train_set}; do
          utils/copy_data_dir.sh  $tgtdir/data$suffix/${datadir} $tgtdir/data$suffix/${datadir}_nosp
          utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_nosp

       done
      elif [ "${wave_sample}" = 16k ];then
       log "Stage 16:  We don't use speed perturb for 16k train data, it is 13 dim mfcc"
        for datadir in ${train_set}; do
          utils/copy_data_dir.sh $tgtdir/data$suffix/${datadir} $tgtdir/data$suffix/${datadir}_nosp
          utils/fix_data_dir.sh $tgtdir/data$suffix/${datadir}_nosp
       done
      else
       log "Error: not supported: --wave-sample ${wave_sample}"
       exit 2
      fi

  fi
fi

if [ "${feats_type}" = perturb ]; then
   lores_train_data_dir=$tgtdir/data$suffix/${datadir}_sp
else
   lores_train_data_dir=$tgtdir/data$suffix/${datadir}_nosp
fi

if [ ${stage} -le 17 ] && [ ${stop_stage} -ge 17 ];then
  if [ "${feats_type}" = perturb ]; then
     if [ "${wave_sample}" = 8k ]; then
        log "Stage 17: using volumn perturb for 8k train data augmentation ,it is 40 dim mfcc"
        for dataset in ${train_set}_sp; do
          utils/copy_data_dir.sh  $tgtdir/data$suffix/$dataset $tgtdir/data$suffix/${dataset}_hires
          utils/data/perturb_data_dir_volume.sh $tgtdir/data$suffix/${dataset}_hires
          steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
           --cmd "$train_cmd" $tgtdir/data$suffix/${dataset}_hires 
          steps/compute_cmvn_stats.sh $tgtdir/data$suffix/${dataset}_hires

          # Remove the small number of utterances that couldn't be extracted for some
          # reason (e.g. too short; no such file).
          utils/fix_data_dir.sh $tgtdir/data$suffix/${dataset}_hires;
       done
     elif [ "${wave_sample}" = 16k ]; then
        log "Stage 17: using volumn perturb for 16k train data augmentation, it is 40 dim mfcc "
        for dataset in ${train_set}_sp; do
          utils/copy_data_dir.sh  $tgtdir/data$suffix/$dataset $tgtdir/data$suffix/${dataset}_hires
          utils/data/perturb_data_dir_volume.sh $tgtdir/data$suffix/${dataset}_hires
          steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
           --cmd "$train_cmd" $tgtdir/data$suffix/${dataset}_hires 
          steps/compute_cmvn_stats.sh $tgtdir/data$suffix/${dataset}_hires

          # Remove the small number of utterances that couldn't be extracted for some
          # reason (e.g. too short; no such file).
          utils/fix_data_dir.sh $tgtdir/data$suffix/${dataset}_hires;
       done
    else
       log "Error: not supported: --wave-sample ${wave_sample}"
       exit 2 
    fi
  else
     if [ "${wave_sample}" = 8k ]; then
        log "Stage 17: We don't use volumn perturb for 8k train data augmentation, it is 40 dim mfcc "
        for dataset in ${train_set}_nosp; do
          utils/copy_data_dir.sh  $tgtdir/data$suffix/$dataset $tgtdir/data$suffix/${dataset}_hires
          steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
           --cmd "$train_cmd" $tgtdir/data$suffix/${dataset}_hires
          steps/compute_cmvn_stats.sh $tgtdir/data$suffix/${dataset}_hires

          # Remove the small number of utterances that couldn't be extracted for some
          # reason (e.g. too short; no such file).
          utils/fix_data_dir.sh $tgtdir/data$suffix/${dataset}_hires;
       done
     elif [ "${wave_sample}" = 16k ]; then
        log "Stage 17: We don't use volumn perturb for 16k train data augmentation, it is 40 dim mfcc "
        for dataset in ${train_set}_nosp; do
          utils/copy_data_dir.sh  $tgtdir/data$suffix/$dataset $tgtdir/data$suffix/${dataset}_hires
          steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
           --cmd "$train_cmd" $tgtdir/data$suffix/${dataset}_hires
          steps/compute_cmvn_stats.sh $tgtdir/data$suffix/${dataset}_hires

          # Remove the small number of utterances that couldn't be extracted for some
          # reason (e.g. too short; no such file).
          utils/fix_data_dir.sh $tgtdir/data$suffix/${dataset}_hires;
       done
     else
       log "Error: not supported: --wave-sample ${wave_sample}"
       exit 2
    fi

  fi

fi

if [ ${stage} -le 18 ] && [ ${stop_stage} -ge 18 ];then
  if [ "${wave_sample}" = 8k ]; then  
    log "Stage 18: get 40 dim mfcc for 8k $test_sets, it is 40 dim mfcc, it is used to get its 100 dim ivector feature"
    for dataset in ${test_sets}; do
       utils/copy_data_dir.sh  $tgtdir/data$suffix/${dataset}_test $tgtdir/data$suffix/${dataset}_test_hires
       steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires_8k.conf \
           --cmd "$train_cmd" $tgtdir/data$suffix/${dataset}_test_hires
       steps/compute_cmvn_stats.sh $tgtdir/data$suffix/${dataset}_test_hires

       # Remove the small number of utterances that couldn't be extracted for some
       # reason (e.g. too short; no such file).
       utils/fix_data_dir.sh $tgtdir/data$suffix/${dataset}_test_hires;
    done
  elif [ "${wave_sample}" = 16k ]; then
    log "Stage 18: get 40 dim mfcc for 16k $test_sets, it is 40 dim mfcc, it is used to get its 100 dim ivector feature"
    for dataset in ${test_sets}; do
       utils/copy_data_dir.sh  $tgtdir/data$suffix/${dataset}_test $tgtdir/data$suffix/${dataset}_test_hires
       steps/make_mfcc.sh --nj 40 --mfcc-config conf/mfcc_hires.conf \
           --cmd "$train_cmd" $tgtdir/data$suffix/${dataset}_test_hires
       steps/compute_cmvn_stats.sh $tgtdir/data$suffix/${dataset}_test_hires

       # Remove the small number of utterances that couldn't be extracted for some
       # reason (e.g. too short; no such file).
       utils/fix_data_dir.sh $tgtdir/data$suffix/${dataset}_test_hires;
    done
  else
    log "Error: not supported: --wave-sample ${wave_sample}"
    exit 2
  fi
fi
# $train_data_sp_hires_dir is 40 dim mfcc , it is used to train ivector extrator
#                                           it is used as input data of chain model.
if [ "${feats_type}" = perturb ]; then
  train_data_sp_hires_dir=$tgtdir/data$suffix/${datadir}_sp_hires
else
  train_data_sp_hires_dir=$tgtdir/data$suffix/${datadir}_nosp_hires
fi

# ivector extractor training
if [ ${stage} -le 19 ] && [ ${stop_stage} -ge 19 ]; then
  log "Stage 19: get pca_transform ....."
  steps/online/nnet2/get_pca_transform.sh --cmd "$cmd" \
    --splice-opts "--left-context=3 --right-context=3" \
    --max-utts 10000 --subsample 2 \
    $train_data_sp_hires_dir \
    $exp_root/nnet3${nnet3_affix}/pca_transform
fi

if [ ${stage} -le 20 ] && [ ${stop_stage} -ge 20 ]; then
  log "Stage 20: train diag ubm ......."
  steps/online/nnet2/train_diag_ubm.sh --cmd "$cmd" --nj 50 --num-frames 700000 \
    $train_data_sp_hires_dir 512 \
    $exp_root/nnet3${nnet3_affix}/pca_transform $exp_root/nnet3${nnet3_affix}/diag_ubm
fi

if [ ${stage} -le 21 ] && [ ${stop_stage} -ge 21 ]; then
  log "Stage 21: train a ivector extractor ......"
  steps/online/nnet2/train_ivector_extractor.sh --cmd "$cmd" --nj 10 \
    $train_data_sp_hires_dir $exp_root/nnet3${nnet3_affix}/diag_ubm \
    $exp_root/nnet3${nnet3_affix}/extractor || exit 1;
fi


if [ "${feats_type}" = perturb ]; then
  train_data_sp_hires=${datadir}_sp_hires
else
  train_data_sp_hires=${datadir}_nosp_hires
fi

if [ ${stage} -le 22 ] && [ ${stop_stage} -ge 22 ]; then
  # We extract iVectors on all the ${train_set} data, which will be what we
  # train the system on.
  # having a larger number of speakers is helpful for generalization, and to
  # handle per-utterance decoding well (iVector starts at zero).
  log "Stage 22: modify speaker info ...."
  utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    $train_data_sp_hires_dir ${train_data_sp_hires_dir}_max2_hires
  log "Step 22: get train set ivector feature......"
  steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 30 \
    ${train_data_sp_hires_dir}_max2_hires $exp_root/nnet3${nnet3_affix}/extractor \
    $exp_root/nnet3${nnet3_affix}/ivectors_${train_data_sp_hires}  || exit 1;
fi


if [ ${stage} -le 23 ] && [ ${stop_stage} -ge 23 ]; then
  log "Stage 23: get 100 dim ivector feature for $test_set"
  for dataset in $test_sets; do
    steps/online/nnet2/extract_ivectors_online.sh --cmd "$cmd" --nj 10 \
      $tgtdir/data$suffix/${dataset}_test_hires $exp_root/nnet3${nnet3_affix}/extractor \
      $exp_root/nnet3${nnet3_affix}/ivectors_${dataset}_test_hires || exit 1;
  done
fi

gmm=tri4
gmm_dir=$exp_root/$gmm
ali_dir=$exp_root/${gmm}_ali${sp_suffix}
if [ $stage -le 24 ] && [ ${stop_stage} -ge 24 ]; then
 if [ "${feats_type}" = perturb ];then
   if [ "${dnn_size}" = small ];then
     log "Stage 24 : aligning with the perturbed low-resolution data, it is used to bulid chain tree, it is used to small model(e.g: tdnnf)"
      steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
        $tgtdir/data$suffix/${train_set}_sp $lang $gmm_dir $ali_dir || exit 1
   elif [ "${dnn_size}" = normal ];then
     log "Stage 24: aligning with the perturbed low-resolution data, it is used to bulid chain tree, it is used to big model(e.g: cnn_tdnnf)"
      steps/align_fmllr.sh --nj 80 --cmd "$train_cmd" \
        $tgtdir/data$suffix/${train_set}_sp $lang $gmm_dir $ali_dir || exit 1
   else
     log "Stage 24: Don't support ${dnn_size}"
   fi
 else
     if [ "${dnn_size}" = small ];then
       log "Stage 24: aligning with the no perturbed  data, it is used to bulid chain tree, it is used to small model(e.g: tdnnf)"
        steps/align_fmllr.sh --nj 20 --cmd "$train_cmd" \
         $tgtdir/data$suffix/${train_set} $lang $gmm_dir $ali_dir || exit 1
     elif [ "${dnn_size}" = normal ];then
       log "Stage 24: aligning with the no perturbed  data, it is used to bulid chain tree, it is used to big model(e.g: cnn_tdnnf)"
         steps/align_fmllr.sh --nj 40 --cmd "$train_cmd" \
          $tgtdir/data$suffix/${train_set} $lang $gmm_dir $ali_dir || exit 1
     else
       log "Stage 24: Don't support ${dnn_type}"
     fi
 fi
fi


test_set_dir=$tgtdir/data$suffix
test_set_ivector_dir=$exp_root/nnet3${nnet3_affix}
if [ ${stage} -le 25 ] && [ ${stop_stage} -ge 25 ];then
 if [ "${feats_type}" = perturb ];then
  if [ "${dnn_size}" = small ];then
     log "Stage 25 train a small cnn_tdnnf chain model with sp, because train data is small, if duration of train data is less 50 hours, it is small."
     source-md/w2020/kaldi-recipe/egs/maison2_nana/run_cnn_tdnnf_small_1a.sh \
       --stage $chain_stage \
       --train-stage $chain_model_train_stage \
       --get-egs-stage -10 \
       --num-epochs  ${num_epochs} \
       --train-set $train_set \
       --gmm tri4 \
       --first-lang $tgtdir/data$suffix/lang \
       --sp-suffix _sp \
       --affix _small_$nnet3_affix  \
       --tree-affix "" \
       --nnet3-affix $nnet3_affix \
       --exp-root $exp_root \
       --tgtdir $tgtdir \
       --suffix $suffix 
  elif [ "${dnn_size}" = normal ];then
    log "Stage 25 train a normal cnn_tdnnf chain model with sp"
    source-md/w2020/kaldi-recipe/egs/maison2_nana/run_cnn_tdnnf_1a.sh \
      --stage $chain_stage \
      --train-stage $chain_model_train_stage \
      --get-egs-stage -10 \
      --num-epochs  ${num_epochs} \
      --train-set $train_set \
      --gmm tri4 \
      --first-lang $tgtdir/data$suffix/lang \
      --sp-suffix _sp \
      --affix _normal_$nnet3_affix  \
      --tree-affix "" \
      --nnet3-affix $nnet3_affix \
      --exp-root $exp_root \
      --tgtdir $tgtdir \
      --suffix $suffix 
  else
     log "Stage 25: Don't support ${dnn_size}"

  fi
 else 
  if [ "${dnn_size}" = small ];then
   log "Stage 25 train a small cnn_tdnnf chain model without sp, because train data is small, if duration of train data is less 50 hours, it is small."
   source-md/w2020/kaldi-recipe/egs/maison2_nana/run_cnn_tdnnf_small_1a.sh \
      --stage $chain_stage \
      --train-stage $chain_model_train_stage \
      --get-egs-stage -10 \
      --num-epochs  ${num_epochs} \
      --train-set $train_set \
      --gmm tri4 \
      --first-lang $tgtdir/data$suffix/lang \
      --sp-suffix _nosp \
      --affix _small_$nnet3_affix \
      --tree-affix "" \
      --nnet3-affix $nnet3_affix \
      --exp-root $exp_root \
      --tgtdir $tgtdir \
      --suffix $suffix 
  elif [ "${dnn_size}" = normal ];then
    log "Stage 25 train a  normal cnn_tdnnf  chain model without sp"
    source-md/w2020/kaldi-recipe/egs/maison2_nana/run_cnn_tdnnf_1a.sh \
      --stage $chain_stage \
      --train-stage $chain_model_train_stage \
      --get-egs-stage -10 \
      --num-epochs  ${num_epochs} \
      --train-set $train_set \
      --gmm tri4 \
      --first-lang $tgtdir/data$suffix/lang \
      --sp-suffix _nosp \
      --affix _normal_$nnet3_affix  \
      --tree-affix "" \
      --nnet3-affix $nnet3_affix \
      --exp-root $exp_root \
      --tgtdir $tgtdir \
      --suffix $suffix 
  else
     log "Stage 25: Don't support ${dnn_size}"
     #log "Stage 25: Error: not supported: --wave-sample ${wave_sample}"
     exit 2
  fi
 fi
fi
if [ ${stage} -le 26 ] && [ ${stop_stage} -ge 26 ];then
 if [ "${dnn_size}" = small ];then
  dir=$exp_root/chain_small_${nnet3_affix}/cnn_tdnn_small_${nnet3_affix}${sp_suffix}   # chain modle output path
 elif [ "${dnn_size}" = normal ];then
  dir=$exp_root/chain_normal_${nnet3_affix}/cnn_tdnn_normal_${nnet3_affix}${sp_suffix}   # chain modle output path
 else
     log "####LOG: chain model output path error"
 fi
 log "chain modle output path is $dir"
fi
#dir=$exp_root/chain${nnet3_affix}/cnn_tdnn${nnet3_affix}${sp_suffix}   # chain modle output path
graph_dir=$dir/graph_better$pp_suffix
if [ ${stage} -le 27 ] && [ ${stop_stage} -ge 27 ];then
  if $use_pp ;then
    log " make HCLG.fst graph with dict_pp...." 
    utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data$suffix/lang_test_pp $dir $graph_dir
  else
    log " make HCLG.fst graph without dict_pp...."
     utils/mkgraph.sh --self-loop-scale 1.0 --remove-oov $tgtdir/data$suffix/local/lang_test $dir $graph_dir
  fi
fi


if [ ${stage} -le 28 ] && [ ${stop_stage} -ge 28 ];then
   log "Stage 28: decode test set offline ...."
   if $use_pp ;then
     for decode_set in $test_sets; do
      #_nj=$(min "${decode_nj}" "$(<" $test_set_dir/${decode_set}_test_hires/utt2spk" wc -l)")
      _nj=${decode_nj}
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $_nj --cmd "$decode_cmd" $iter_opts \
          --skip_scoring true \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_test_hires \
          $graph_dir $test_set_dir/${decode_set}_test_hires \
          $dir/decode_${decode_set}_test_better_pp

     done
   wait

   else
     for decode_set in $test_sets; do
      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      # if decode_nj is very big(e.g: 2716) , it may decode incorrectly, you should reduce nj.
      #_nj=$(min "${decode_nj}" "$(<" $test_set_dir/${decode_set}_test_hires/utt2spk" wc -l)")
      #decode_nj=$(wc -l $tgtdir/data/${decode_set}_hires/spk2utt | awk '{print $1}' || exit 1;)
      _nj=${decode_nj}
      steps/nnet3/decode.sh --acwt 1.0 --post-decode-acwt 10.0 \
          --nj $_nj --cmd "$decode_cmd" $iter_opts \
          --skip_scoring true \
          --online-ivector-dir $test_set_ivector_dir/ivectors_${decode_set}_test_hires \
          $graph_dir $test_set_dir/${decode_set}_test_hires \
          $dir/decode_${decode_set}_test_better || exit 1;

   done
   wait
   fi
fi

if [ ${stage} -le 29 ] && [ ${stop_stage} -ge 29 ];then
 log " Stage 29 : make wer score for test set..."
 if $use_pp ;then
   for decode_set in $test_sets; do
    source-md/analaysis/analysis_asr_results/score.sh \
       --cmd "$decode_cmd" \
      $test_set_dir/${decode_set}_test_hires \
      $graph_dir  \
      $dir/decode_${decode_set}_test_better_pp
   done
 else
   for decode_set in $test_sets; do
    source-md/analaysis/analysis_asr_results/score.sh \
       --cmd "$decode_cmd" \
      $test_set_dir/${decode_set}_test_hires \
      $graph_dir  \
      $dir/decode_${decode_set}_test_better
   done

  fi 
fi

if [ ${stage} -le 30 ] && [ ${stop_stage} -ge 30 ];then
   log " Stage 30 : show wer score for test set..."
   [ -f $dir/RESULTS.md ] && rm $dir/RESULTS.md 
   if $use_pp ;then
     for decode_set in $test_sets; do
         head $dir/decode_${decode_set}_test_better_pp/scoring_kaldi/best_wer >>$dir/RESULTS.md 
     done
   else
     for decode_set in $test_sets; do
         head $dir/decode_${decode_set}_test_better/scoring_kaldi/best_wer >> $dir/RESULTS.md
                                                       
     done
    
  fi
  cat $dir/RESULTS.md
fi
log "Successfully finished. [elapsed=${SECONDS}s]"
