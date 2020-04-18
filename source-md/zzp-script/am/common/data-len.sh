#!/bin/bash

data=$1

if [ $# -ne 1 ] || [ ! -f $data/segments ]; then
  echo "segments file expected from data $data"
  exit 1
fi

echo "data length:"
cat $data/segments | \
awk '{x+=$4-$3;}END{print x/3600;}'
