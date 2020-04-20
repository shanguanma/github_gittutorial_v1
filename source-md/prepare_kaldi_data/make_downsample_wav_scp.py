#!/usr/bin/env python3
# encoding: utf-8

# Copyright 2019 Ma Duo
# Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse


def get_parse():
    parser = argparse.ArgumentParser(
        description="make downsample wav.scp, e.g: from 16k-->8k"
    )
    parser.add_argument(
        "wav_scp", type=str, help="input wav.scp file path, it may be 16k"
    )
    args = parser.parse_args()
    return args


def main():
    args = get_parse()
    with open(args.wav_scp, "r", encoding="utf-8") as f:
        wav_scp = f.readlines()
    for line in wav_scp:
        line = line.strip().split(" ")
        path = line[1].split("/")
        suffix = path[-1].split(".")[-1]
        if suffix == "wav":
            # wav format
            # example: /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/train_imda_part3_ivr/wav.scp
            print(
                "{0} /usr/bin/sox -t wav {1} -r 8000 -c 1 -b 16 -t wav - |".format(
                    line[0], line[1]
                )
            )
        else:
            # suffix == 'pcm':
            # pcm format
            print(
                "{0} /usr/bin/sox  {1} -r 8000 -c 1 -b 16 -t wav - |".format(
                    line[0], line[1]
                )
            )

        """
        # I modified sg-en-i2r.
        # c01-spkr201-f21-ch-mob-son-03 /usr/bin/sox -r 8000 /data/users/hhx502/w2016/sg-en-i2r/CategoryI/Wave/Speaker201/Speaker201-3.pcm -r 16000 -c 1 -b 16 -t wav - |
        print('{0} /usr/bin/sox {1} -r 8000 -c 1 -b 16 -t wav - |'.format(line[0], line[4])) 
        """


if __name__ == "__main__":
    main()

# how to run?
# source-md/prepare_kaldi_data/make_downsample_wav_scp.py  test/MSF_Baby_Bonus_Transcript_wav_scp_5 > test/MSF_Baby_Bonus_Transcript_wav_scp_5_downsample_to_8k
# ivr-conf_2501_2501_00862070 sox -t wav /data/users/ngaho/corpora/IMDA-NSC-Part3/PART3/Audio_Separate_IVR/conf_2501_2501/00862070.wav  -r 8000 -c 1 -b 16 -t wav - |

# c01-spkr201-f21-ch-mob-son-03 /usr/bin/sox -r 8000 /data/users/hhx502/w2016/sg-en-i2r/CategoryI/Wave/Speaker201/Speaker201-3.pcm -r 16000 -c 1 -b 16 -t wav - |
