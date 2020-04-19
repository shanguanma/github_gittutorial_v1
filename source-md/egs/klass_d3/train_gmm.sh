#!/bin/bash

. path.sh

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

 $0 --cmd slurm.pl --nj 17 --steps 1,2,3,4,5,6,7 \
 --train-id a --cmvn-opts "--norm-means=true"  --state-num 5000 --pdf-num 75000 \
 --devdata ~/w2016/kws2016/flp-grapheme-phone/data/dev/plp_pitch \
 ~/w2016/kws2016/flp-grapheme-phone/data/train/plp_pitch ~/w2016/kws2016/flp-grapheme-phone/data/lang \
 ~/w2016/kws2016/flp-grapheme-phone/exp
 
 $0 --cmd slurm.pl --nj 40 --steps 1,2,3,4,5,6,7  --train-id b --cmvn-opts "--norm-means=true" \
 --state-num 5000 --pdf-num 100000 \
 --devdata /home2/hhx502/ted-libri-en/data/dev_tedlium_r1-3m/mfcc-pitch \
 /home2/hhx502/ted-libri-en/data/train-merge-ted12-libri460/sub0.3-mfcc-pitch \
 /home2/hhx502/ted-libri-en/data/g2p-libri/lang /home2/hhx502/ted-libri-en/exp

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
  echo "Mono training started @ `date`"
  if [ ! -f $mono_dir/final.mdl ]; then
    steps/train_mono.sh --cmd "$cmd" --nj $nj ${cmvn_opts:+--cmvn-opts "$cmvn_opts"} \
    --totgauss $mono_gauss  $data  $lang $mono_dir || exit 1
  fi
  steps/align_si.sh   --nj $nj --cmd "$cmd" $data  $lang $mono_dir  $mono_ali || exit 1
  echo "Mono training ended @ `date`"
fi

tri1_dir=$rdir/tri1$train_id
tri1_ali=$tri1_dir/ali_${dataname}
if [ ! -z $step02 ]; then
  echo "delta feature based tri training started @ `date`"
  steps/train_deltas.sh  --cmd "$cmd" ${cmvn_opts:+--cmvn-opts "$cmvn_opts"} \
  $state_num $pdf_num  $data  $lang $mono_ali $tri1_dir || exit 1
  steps/align_si.sh  --cmd "$cmd" --nj $nj  $data  $lang $tri1_dir  $tri1_ali || exit 1
  echo "Done @ `date`"
fi
[ ! -z $alidir ] && tri1_ali=$alidir
tri2_dir=$rdir/tri2$train_id
tri2_ali=$tri2_dir/ali_${dataname}
if [ ! -z $step03 ]; then
  echo "lda+mllt training started @ `date`"
  steps/train_lda_mllt.sh  --cmd "$cmd" --dim $lda_dim  ${cmvn_opts:+--cmvn-opts "$cmvn_opts"} \
  $state_num $pdf_num $data  $lang $tri1_ali $tri2_dir || exit 1
  steps/align_si.sh --cmd "$cmd" --nj $nj \
  $data  $lang $tri2_dir  $tri2_ali || exit 1
  echo "Done @ `date`"
fi

if $done_with_lda_mllt; then
  echo "## LOG ($0): we are done with lda_mllt training"
  exit 0
fi

tri3_dir=$rdir/tri3$train_id
tri3_ali=$tri3_dir/ali_${dataname}
[ ! -z $alidir ] && tri2_ali=$alidir
if [ ! -z $step04 ]; then
  echo "SAT training started @ `date`"
  steps/train_sat.sh  --cmd "$cmd"  $state_num $pdf_num \
  $data  $lang $tri2_ali $tri3_dir || exit 1
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $data $lang $tri3_dir $tri3_ali || exit 1
  echo "Done @ `date`"
fi
tri4_dir=$rdir/tri4$train_id
tri4_ali=$tri4_dir/ali_$dataname
if [ ! -z $step05 ]; then
  echo "SAT training started @ `date`"
  steps/train_sat.sh  --cmd "$cmd"  $state_num $pdf_num \
  $data  $lang $tri3_ali $tri4_dir || exit 1
  steps/align_fmllr.sh --cmd "$cmd" --nj $nj \
  $data $lang $tri4_dir $tri4_ali || exit 1
  echo "Done @ `date`"
fi
graphdir=$tri4_dir/$graphname
srcdir=$tri4_dir
if [ -z $step05 ]; then  
  graphdir=$tri3_dir/$graphname; srcdir=$tri3_dir 
fi
if [ ! -z $step06 ]; then
  echo "Making graph started @ `date`"
  [ -f $srcdir/final.mdl ] || \
  { echo "ERROR, final.mdl expected from srcdir $srcdir"; exit 1; }
  if [ ! -f $graphdir/HCLG.fst ]; then
    utils/mkgraph.sh $lang $srcdir  $graphdir
  fi
  echo "Done @ `date`"
fi

if [ ! -z $step07 ]; then
  echo "Decoding started @ `date`"
  effect_nj=$(wc -l < $devdata/spk2utt)
  [ $effect_nj -gt $nj ] && effect_nj=$nj
  [ -f $graphdir/HCLG.fst ] || \
  { echo "ERROR, HCLG.fst expected from $graphdir"; exit 1; }
  $decode_cmd --cmd "$cmd" --nj $effect_nj \
  --scoring-opts "$scoring_opts" \
  $decode_opts $graphdir $devdata $srcdir/$decodename || exit 1
  echo "Decoding Done @ `date`"
fi

