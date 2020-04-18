#!/bin/bash 

. path.sh
. env.sh


# first copy data to the target folder
./scripts/copy-data-by-adding-codec.sh --steps 1 --spk-prefix 'codec-good-' --utt-prefix 'codec-good-' \
 /home3/zpz505/w2019/seame/data/train \
 ./codec-list-good.txt \
 ./exp/seame-train-good-codecs

# now rewrite the wav.scp according to the segments
./scripts/copy-data-by-adding-codec.sh --steps 2 --spk-prefix 'codec-good-' --utt-prefix 'codec-good-' \
 /home3/zpz505/w2019/seame/data/train \
 ./codec-list-good.txt \
 ./exp/seame-train-good-codecs

# do codec processing on wav.scp
./scripts/copy-data-by-adding-codec.sh --steps 3 --spk-prefix 'codec-good-' --utt-prefix 'codec-good-' \
 /home3/zpz505/w2019/seame/data/train \
 ./codec-list-good.txt \
 ./exp/seame-train-good-codecs

# make wav2dur and then prepare new segments file
# Can check the progress by these commands
#   -> to see the number of jobs finished
#     grep Ended ./exp/seame-train-good-codecs/log/get_reco_durations.* | wc -l 
#
#   -> to see the number of jobs finished successfully
#     grep Ended ./exp/seame-train-good-codecs/log/get_reco_durations.* | grep 'code\s0' | wc -l
#  
./scripts/copy-data-by-adding-codec.sh --steps 4 --spk-prefix 'codec-good-' --utt-prefix 'codec-good-' \
 --cmd "slurm.pl --quiet" --nj 1000 \
 /home3/zpz505/w2019/seame/data/train \
 ./codec-list-good.txt \
 ./exp/seame-train-good-codecs

# make a trivial segments file
./scripts/copy-data-by-adding-codec.sh --steps 5 --spk-prefix 'codec-good-' --utt-prefix 'codec-good-' \
 /home3/zpz505/w2019/seame/data/train \
 ./codec-list-good.txt \
 ./exp/seame-train-good-codecs


