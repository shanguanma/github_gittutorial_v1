#!/bin/bash

if [ $# -ne 1 ]; then
  echo
  echo  "$0 <decode-dir>"
  echo
  exit 1
fi

decode_dir=$1

grep Sum $decode_dir/scor*/*.sys |perl -pe 's/.*(score_\d+.*)/$1/g;s/[:|]//g;' | sort -k9n

#sort -k3n means sort by the number in the third column.
grep WER $decode_dir/wer_* | perl -pe 's/:/ /g;' |  sort -k3n
