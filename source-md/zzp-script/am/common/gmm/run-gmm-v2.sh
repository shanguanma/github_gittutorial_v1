#!/bin/bash

. path.sh
. cmd.sh

# begin options
cmd=slurm.pl
nj=40
steps=
train_id=a
dataname=train
cmvn_opts=
mono_gauss=2500
state_num=2500
pdf_num=40000
lda_dim=40
alidir=
done_with_lda_mllt=false

decode_cmd=steps/decode_fmllr.sh
devdata=
devname=dev
graphname=graph
decodename=decode_dev
decode_opts="--acwt 0.1 --skip-scoring false"
scoring_opts="--min-lmwt 8 --max-lmwt 25"

do_subset_mono=false
subset_mono=300000
do_subset_tri12=false
subset_tri12=900000

stage_mono=-10
#stage_mono_ali=-10
stage_tri1=-10
#stage_tri1_ali=-10
stage_tri2=-10
#stage_tri2_ali=-10
stage_tri3=-10
stage_tri3_ali=-10
stage_tri4=-10
stage_tri4_ali=-10

dict=
generate_ali_from_lats=false # If true, Tri4 alingments generated from lattices.
# end options
echo 
echo "LOG: $0 $@"
echo

. parse_options.sh || exit 1

function Usage {
  cat<<END
 
 $(basename $0) [options] <data> <lang> <tgtdir>
 [options]:
 --cmd                            # value, "$cmd"
 --nj                             # value, $nj
 --steps                          # value, "$steps", for instance, "--steps 1,2,3,4"
 --train-id                       # value, "$train_id"
 --dataname                       # value, "$dataname"
 --cmvn-opts                      # value, "$cmvn_opts"
 --mono-gauss                     # value, $mono_gauss
 --state-num                      # value, $state_num
 --pdf-num                        # value, $pdf_num
 --lda-dim                        # value, $lda_dim
 --alidir                         # value, "$alidir"
 --done-with-lda-mllt             # value, "$done_with_lda_mllt"
 --do-subset-mono                 # value, "$do_subset_mono"
 --subset-mono                    # value, "$subset_mono"
 --do-subset-tri12                # value, "$do_subset_tri12"
 --subset-tri12                   # value, "$subset_tri12"

 --decode-cmd                     # value, "$decode_cmd"
 --devdata                        # value, "$devdata"
 --devname                        # value, "$devname"
 --graphname                      # value, "$graphname"
 --decodename                     # value, "$decodename"
 --decode-opts                    # value, "$decode_opts"
 --scoring-opts                   # value, "$scoring_opts"

 [steps]:
 1: train mono
 2: train delta tri
 3: train lda mllt
 4: train sat
 5: train sat again

 6: make decoding graph
 7: decoding on dev data, if any

 [examples]:

 source/egs/swahili/run-gmm.sh --cmd slurm.pl --steps 2 --pdfs 5000 --mixtures 75000 --devdata flp-grapheme/data/dev/plp_pitch \
 flp-grapheme/data/train/plp_pitch flp-grapheme/data/lang flp-grapheme/exp/mono

 train GMM default --steps 1-11  1.mono 2.ali 3. tri1 4.ali  5.tri2 6.ali  7.tri3 8.ali 9.lang-silprob 10.tri4 11.ali

END
}

if [ $# -ne 3 ]; then
  Usage && exit 1;
fi

data=$1
lang=$2
rdir=$3

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

if [ ! -z $alidir ]; then
  step01=
  step02=
  step03=
  step05=
fi





mono_dir=$rdir/mono0$train_id
mono_ali=$mono_dir/ali_${dataname}
if [ ! -z $step01 ]; then
  if $do_subset_mono; then
   num=$(wc -l < $data/utt2spk)
   [ $num -lt $subset_mono ] && subset_mono=$num
   utils/subset_data_dir.sh --speakers $data $subset_mono $data/../train_100kshort || exit 1
   monodata=$data/../train_100kshort || exit 1
  else
     monodata=$data || exit 1
  fi
  echo "Mono training(step01) started @ `date`"
#  if [ ! -f $mono_dir/final.mdl ]; then
    steps/train_mono.sh --cmd "$cmd" --nj $nj ${cmvn_opts:+--cmvn-opts "$cmvn_opts"} --stage $stage_mono \
    --totgauss $mono_gauss  $monodata  $lang $mono_dir || exit 1
#  fi
    echo "Mono training(step01) ended @ `date`"
fi

is_subset_tri12_now=false
if [ ! -z $step02 ]; then
  if $do_subset_tri12 && ! $is_subset_tri12_now ; then
    num=$(wc -l < $data/utt2spk)
    [ $num -lt $subset_tri12 ] && subset_tri12=$num
    utils/subset_data_dir.sh --speakers $data $subset_tri12 $data/../train_300kshort || exit 1
    tridata=$data/../train_300kshort || exit 1
    is_subset_tri12_now=true
  elif $is_subset_tri12_now ; then
    tridata=$data/../train_300kshort
  else
    tridata=$data || exit 1
  fi
  steps/align_si.sh   --nj $nj --cmd "$cmd" $tridata  $lang $mono_dir  $mono_ali || exit 1
  echo "Mono ali(step02) ended @ `date`"
fi

tri1_dir=$rdir/tri1$train_id
tri1_ali=$tri1_dir/ali_${dataname}
if [ ! -z $step03 ]; then
  if $do_subset_tri12 && ! $is_subset_tri12_now ; then
    num=$(wc -l < $data/utt2spk)
    [ $num -lt $subset_tri12 ] && subset_tri12=$num
    utils/subset_data_dir.sh --speakers $data $subset_tri12 $data/../train_300kshort || exit 1
    tridata=$data/../train_300kshort || exit 1
    is_subset_tri12_now=true
  elif $is_subset_tri12_now ; then
    tridata=$data/../train_300kshort
  else
    tridata=$data || exit 1
  fi
  echo "Delta feature based tri(step03) training started @ `date`"
  steps/train_deltas.sh  --cmd "$cmd" --stage $stage_tri1  ${cmvn_opts:+--cmvn-opts "$cmvn_opts"} \
  $state_num $pdf_num  $tridata  $lang $mono_ali $tri1_dir || exit 1
  echo "Tri1 training(step03) ended @ `date`"
fi

if [ ! -z $step04 ]; then
  if $do_subset_tri12 && ! $is_subset_tri12_now ; then
    num=$(wc -l < $data/utt2spk)
    [ $num -lt $subset_tri12 ] && subset_tri12=$num
    utils/subset_data_dir.sh --speakers $data $subset_tri12 $data/../train_300kshort || exit 1
    tridata=$data/../train_300kshort || exit 1
    is_subset_tri12_now=true
  elif $is_subset_tri12_now ; then
    tridata=$data/../train_300kshort
  else
    tridata=$data || exit 1
  fi
  steps/align_si.sh  --cmd "$cmd" --nj $nj  $tridata  $lang $tri1_dir  $tri1_ali || exit 1
  echo "Tri1 ali(step04) ended @ `date`"
fi

[ ! -z $alidir ] && tri1_ali=$alidir
tri2_dir=$rdir/tri2$train_id
tri2_ali=$tri2_dir/ali_${dataname}
if [ ! -z $step05 ]; then
  if $do_subset_tri12 && ! $is_subset_tri12_now ; then
    num=$(wc -l < $data/utt2spk)
    [ $num -lt $subset_tri12 ] && subset_tri12=$num
    utils/subset_data_dir.sh --speakers $data $subset_tri12 $data/../train_300kshort || exit 1
    tridata=$data/../train_300kshort || exit 1
    is_subset_tri12_now=true
  elif $is_subset_tri12_now ; then
    tridata=$data/../train_300kshort
  else
    tridata=$data || exit 1
  fi
  echo "lda+mllt training(step05) started @ `date`"
  steps/train_lda_mllt.sh  --cmd "$cmd" --stage $stage_tri2 --dim $lda_dim  ${cmvn_opts:+--cmvn-opts "$cmvn_opts"} \
  $state_num $pdf_num $tridata  $lang $tri1_ali $tri2_dir || exit 1
  echo "Tri2 training(step05) ended @ `date`"
fi

if [ ! -z $step06 ]; then
  steps/align_si.sh --cmd "$cmd" --nj $nj \
  $data  $lang $tri2_dir  $tri2_ali || exit 1
  echo "Tri2 ali(step06) ended @ `date`"
fi

if $done_with_lda_mllt; then
  echo "## LOG ($0): we are done with lda_mllt training"
  exit 0
fi

tri3_dir=$rdir/tri3$train_id
tri3_ali=$tri3_dir/ali_${dataname}
[ ! -z $alidir ] && tri2_ali=$alidir
if [ ! -z $step07 ]; then
  echo "SAT training(step07) started @ `date`"
  steps/train_sat.sh  --cmd "$cmd" --stage $stage_tri3 $state_num $pdf_num \
  $data  $lang $tri2_ali $tri3_dir || exit 1
  echo "Tri3 training(step07) ended @ `date`"
fi
if [ ! -z $step08 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj --stage $stage_tri3_ali \
  $data $lang $tri3_dir $tri3_ali || exit 1
  echo "Tri3 ali(step08) ended @ `date`"
fi

if [ ! -z $dict ]; then
  #dict=$lang/../local/dict #default located
  olddictdir=$dict
  dictdir=${dict}-silprob
  oldlang=$lang
  lang=${lang}-silprob
  if [ ! -z $step09 ]; then
    echo "Update lang for tri4 (step09) started @ `date`"
    steps/get_prons.sh --cmd "$cmd"  $data $oldlang $tri3_ali   || exit 1;
    utils/dict_dir_add_pronprobs.sh --max-normalize true \
    $olddictdir $tri3_ali/pron_counts_nowb.txt $tri3_ali/sil_counts_nowb.txt \
    $tri3_ali/pron_bigram_counts_nowb.txt $dictdir   || exit 1;
    [[ -f "$dictdir/lexiconp.txt" ]] && rm $dictdir/lexiconp.txt
    utils/validate_dict_dir.pl $dictdir   || exit 1;
    utils/prepare_lang.sh $dictdir "<unk>" $lang/tmp $lang   || exit 1;
    [ -f $oldlang/G.fst ] && cp $oldlang/G.fst $lang/
    echo "Update lang for tri4 (step09) ended @ `date`"
  fi
fi

tri4_dir=$rdir/tri4$train_id
tri4_ali=$tri4_dir/ali_$dataname
if [ ! -z $step10 ]; then
  echo "SAT training(step10) started @ `date`"
  steps/train_sat.sh  --cmd "$cmd"  --stage $stage_tri4 $state_num $pdf_num \
  $data  $lang $tri3_ali $tri4_dir || exit 1
  echo "Tri4 training(step10) ended @ `date`"
fi

if [ ! -z $step11 ]  && $generate_ali_from_lats; then
  echo "Tri4 lats and generate ali from lats(step11) start @ `date`"
  steps/align_fmllr_lats.sh --cmd "$cmd" --nj $nj --stage $stage_tri4_ali --generate_ali_from_lats $generate_ali_from_lats \
  $data $lang $tri4_dir $tri4_ali || exit 1
  echo "Tri4 lats and generate ali from lats(step11) ended @ `date`"
elif [ ! -z $step11 ]; then
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj --stage $stage_tri4_ali \
  $data $lang $tri4_dir $tri4_ali || exit 1
  echo "Tri4 ali(step11) ended @ `date`"
fi
