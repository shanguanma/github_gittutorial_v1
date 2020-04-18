#!/bin/bash

data=$1

if [ $# -ne 1 ] ||  [ ! -f $data/segments ] && [ ! -f $data/reco2dur ]; then
  echo "segments or reco2dur file expected from data '$data'"
  exit 1
fi

echo "data length:"
[ -f $data/segments ] && \
cat $data/segments | \
awk '{x+=$4-$3;}END{print x/3600;}'  

[ -f $data/reco2dur ] && \
{  cat $data/reco2dur | \
  awk '{x+=$2;} END{print x/3600;}';  }
