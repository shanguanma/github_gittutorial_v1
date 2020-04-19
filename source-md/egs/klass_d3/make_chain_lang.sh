#!/bin/bash 

. path.sh 

echo
echo "## LOG ($0) $@"
echo

if [ $# -ne 2 ]; then
  echo
  echo "Example: $0 source-lang chain_lang"
  echo && exit 1
fi

source_lang=$1
lang=$2
[ -d $lang ] || mkdir -p $lang
cp -rL $source_lang/* $lang/

silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1; 
steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo || exit 1

echo "# done with '$lang'"
exit 0
