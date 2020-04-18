export KALDI_ROOT=/home3/md510/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh
export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$PWD:$PATH
# [  -f $KALDI_ROOT/tools/config/common_path.sh ] 
. $KALDI_ROOT/tools/config/common_path.sh  || echo "file $KALDI_ROOT/tools/config/common_path.sh expected"
export LC_ALL=C
