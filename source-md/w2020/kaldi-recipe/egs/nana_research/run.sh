#!/bin/bash 

. path.sh
. cmd.sh

# dataset is from /data/users/nana511/vctk
# wave path:/data/users/nana511/vctk/wav_fake16/
# text path:/data/users/nana511/vctk/txt

cmd="slurm.pl  --quiet --exclude=node05,node06,node07,node08"
steps=
nj=40
. utils/parse_options.sh || exit 1


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

# prepared kaldi data
data_root_folder_dir=/data/users/nana511/vctk
tgt_data_dir=`pwd`/data/wav_fake16
if [ ! -z $step01 ];then
   mkdir -p $tgt_data_dir
   find $data_root_folder_dir/wav_fake16  -name "*.wav" > $tgt_data_dir/wav_list
   cat  $tgt_data_dir/wav_list | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;print "$label $_\n";'| sort -u> $tgt_data_dir/wav.scp

fi
if [ ! -z $step02 ];then
   #  make utt2spk spk2utt
   cat $tgt_data_dir/wav.scp | awk '{print $1, $1;}' > $tgt_data_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $tgt_data_dir/utt2spk > $tgt_data_dir/spk2utt
fi
if [ ! -z $step03 ];then
   # make text
   find $data_root_folder_dir/txt -name "*.txt"> $tgt_data_dir/txt_list
   cat $tgt_data_dir/txt_list | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.txt::g; $label =~ s:\-mp[34]$::g;  print "$label $_\n";'  > $tgt_data_dir/text_scp

   # corvet text_scp to kaldi text
   source-md/egs/nana_research/make_text.py $tgt_data_dir/text_scp $tgt_data_dir/pre_text
   # remove punctuation in $data_root_folder_dir/pre_text
   sed --regexp-extended 's/[,|\.|?|)|"|!]//g' $tgt_data_dir/pre_text | sed 's/[A-Z]/\l&/g' | sort >$tgt_data_dir/text
fi


# make lang
# I use imda dict, may be is OK
dict=data/dict_imda
lang=data/lang
exp_root=exp
train_set=wav_fake16
if [ ! -z $step04 ];then
   cp -r /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/dict_imda data/
   utils/validate_dict_dir.pl $dict || { echo "## ERROR (step01): failed to validating dict '$dict'" && exit 1;  }
   utils/prepare_lang.sh $dict "<unk>" $lang/tmp $lang 
fi
# make 13 mfcc 
mfccdir=mfcc
if [ ! -z $step05 ];then
  for sdata in $train_set; do
    # this must be make_mfcc ,it shouldn't  add pitch. otherwise running steps/align_fmllr_lats.sh in step18 local/semisup/chain/run_tdnn.sh is error. 
    # beacuse train_aug folder only contain utt2uniq  spk2utt  text  utt2spk  wav.scp,  so must set -write-utt2num-frames is false 
    # beacuse this doesn't use energy(在conf/mfcc.conf可以看到), so　mfcc features is 13  MFCCs, so mfcc dimension is 13
    # if using energy, so mfcc features is 12 MFCCs + Energy, so mfcc dimension is 13
    steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
      --mfcc-config conf/mfcc.conf  --write-utt2num-frames false data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    #This script will fix sorting errors and will remove any utterances for which some required data, such as feature data or transcripts, is missing.
    utils/fix_data_dir.sh data/${sdata}
    echo "## LOG : done with mfcc feat"

  done

fi 

# the gmm-hmm is from /home4/md510/w2019a/kaldi-recipe/add-noise-to-seame-v7/run-reverb-music-noise-to-seame.sh 
if [ ! -z $step06 ];then
  #utils/subset_data_dir.sh --shortest data/train_sup 100000 data/train_sup_100kshort
  utils/subset_data_dir.sh  data/$train_set 10000 data/${train_set}_10k
  utils/data/remove_dup_utts.sh 100 data/${train_set}_10k data/${train_set}_10k_nodup

fi

if [ ! -z $step07 ];then
  steps/train_mono.sh --nj  $nj  --cmd "$cmd" \
    data/${train_set}_10k_nodup $lang $exp_root/mono0a || exit 1
fi

#if [ ! -z $step09 ]; then
if [ ! -z $step08 ];then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    data/$train_set $lang $exp_root/mono0a $exp_root/mono0a_ali || exit 1
  # GMM 的声学特征共 39 维(∆ 表示 delta):
  # if using energy
  # 12 MFCCs + Energy;
  # 12 ∆ MFCCs + ∆ Energy;
  # 12 ∆ 2 MFCCs + ∆ 2 Energy 
  # this doesn't using energy, so 
  # 13 MFCCs
  # 13 ∆ MFCCs
  # 13 ∆ 2 MFCCs
  steps/train_deltas.sh --cmd "$cmd" \
    2500 20000 data/$train_set $lang $exp_root/mono0a_ali $exp_root/tri1 || exit 1
fi
if [ ! -z $step09 ];then

  # 使用最大后验概率估计(Maximum a Posterior, MAP)，进行声学模型自适应调整.
  # 如果可以将自适应语音中的每个语音帧合理地分配到所属的HMM模型.
  # HMM状态以及该状态的某一个高斯成份,我们将可以基于每个高斯成份所对应的语音帧
  # 对该高斯成份的参数进行调整。基于同样的准则,实验表明,在均值、协方差、权重这三
  # 类参数中,对均值的调整效果最为明显,因此,我们将只考虑对均值的更新。
  # 在MAP自适应方法中,对每个高斯成分的自适应是独立进行的。
  # 这一独立更新方案
  # 使得MAP非常灵活;另一方面,由于不同音素、状态、高斯成分之间无法共享数据,使得
  # 那些没有分配到自适应数据的高斯成分无法更新。
  steps/align_si.sh --nj $nj --cmd "$cmd" \
   data/$train_set $lang $exp_root/tri1 $exp_root/tri1_ali || exit 1;
  # 对 MFCC 特征进行了基于线性判别分析 (Linear Discriminant Analysis,LDA)
  # 和最大似然线性变换(Maximum Likelihood Linear Transform,MLLT)的转换
  steps/train_lda_mllt.sh --cmd "$cmd" \
    2500 20000 data/$train_set $lang $exp_root/tri1_ali $exp_root/tri2 || exit 1;

fi

#if [ ! -z $step12 ]; then
if [ ! -z $step10 ];then
  steps/align_si.sh --nj $nj --cmd "$cmd" \
    data/$train_set $lang $exp_root/tri2 $exp_root/tri2_ali || exit 1;
  # 对 MFCC 特征进行了基于线性判别分析 (Linear Discriminant Analysis,LDA) 
  # 和最大似然线性变换(Maximum Likelihood Linear Transform,MLLT)的转换
  steps/train_lda_mllt.sh --cmd "$cmd" \
     --splice-opts "--left-context=3 --right-context=3" \
     5000 40000 data/$train_set $lang $exp_root/tri2_ali $exp_root/tri3a || exit 1;
fi

if [ ! -z $step11 ];then
  # 一种解决方法是设计一个对所有高斯成
  # 分进行统一更新的变换M,使得任何一个高斯成分上分配到的自适应数据都可以对全体高
  # 斯成分产生影响.
  steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    data/$train_set $lang $exp_root/tri3a $exp_root/tri3a_ali || exit 1;
  # speaker adapation train(SAT)
  # fMLLR可以用来训练通用说话人模型。这一方法假设一个人的发音可以通过一个fMLLR从
  # 说话人特征中去掉说话人相关信息,再利用该“中性”特征训练一个通用说话人模型。基
  # 于该中性模型,可再次更新每个人的fMLLR,得到新的中性特征。上述过程可迭代进行
  # 称为SAT训练(Speaker Adaptive Training,SAT)[14]。中性模型不包含说
  # 话人信息,因此可实现更好的建模。在实际进行识别时,首先需要基于中性模型计算个人
  # 的fMLLR,再将该fMLLR应用到待识别语音的特征向量,并基于中性模型进行识别。
  steps/train_sat.sh --cmd "$cmd" \
    5000 100000 data/$train_set $lang $exp_root/tri3a_ali $exp_root/tri4a || exit 1;

fi

if [ ! -z $step12 ];then
   source-md/w2020/kaldi-recipe/egs/nana_research/run_cnn_tdnnf_1a.sh \
    --cmd "$cmd" \
    --train-set $train_set \
    --nnet3-affix _1a \
    --tdnn-affix _1a --tree-affix bi_a \
    --gmm tri4a  \
    --stage 13  \
    --train-stage 0 || exit 1; #`
fi

if [ ! -z $step13 ];then
   steps/align_fmllr.sh --nj $nj --cmd "$cmd" \
    data/$train_set $lang $exp_root/tri4a $exp_root/tri4a_ali || exit 1;

fi



# prepared kaldi data
data_root_folder_dir=/data/users/nana511/vctk/vctk_vad/corpus_1s
tgt_data_dir=`pwd`/data/vctk_wav_fake16
tgt_data_dir_1=`pwd`/data/vctk_wav_fake16_novad_1s_tt
if [ ! -z $step14 ];then
   mkdir -p $tgt_data_dir
   find $data_root_folder_dir/wav_fake16  -name "*.wav" > $tgt_data_dir/wav_list
   cat  $tgt_data_dir/wav_list | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;print "$label $_\n";'| sort -u> $tgt_data_dir/wav.scp

   mkdir -p $tgt_data_dir_1
   find $data_root_folder_dir/wav_fake16_novad_1s_tt  -name "*.wav" > $tgt_data_dir_1/wav_list
   cat  $tgt_data_dir_1/wav_list | \
   perl -ane 'chomp; $label = $_; $label =~ s:.*\/::g;  $label =~ s:\.wav::g; $label =~ s:\-mp[34]$::g;print "$label $_\n";'| sort -u> $tgt_data_dir_1/wav.scp
    


fi
if [ ! -z $step15 ];then
   #  make utt2spk spk2utt
   cat $tgt_data_dir/wav.scp | awk '{print $1, $1;}' > $tgt_data_dir/utt2spk
   utils/utt2spk_to_spk2utt.pl $tgt_data_dir/utt2spk > $tgt_data_dir/spk2utt

    #  make utt2spk spk2utt
   cat $tgt_data_dir_1/wav.scp | awk '{print $1, $1;}' > $tgt_data_dir_1/utt2spk
   utils/utt2spk_to_spk2utt.pl $tgt_data_dir_1/utt2spk > $tgt_data_dir_1/spk2utt
fi

# make 13 dim mfcc
if [ ! -z $step16 ];then
  
  for sdata in vctk_wav_fake16 vctk_wav_fake16_novad_1s_tt; do   
     steps/make_mfcc.sh --cmd "$cmd" --nj $nj \
      --mfcc-config conf/mfcc.conf  --write-utt2num-frames false data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    steps/compute_cmvn_stats.sh data/${sdata} exp/make_mfcc/${sdata} $mfccdir || exit 1;
    #This script will fix sorting errors and will remove any utterances for which some required data, such as feature data or transcripts, is missing.
    utils/fix_data_dir.sh data/${sdata}
    echo "## LOG : done with mfcc feat"

  done

fi
