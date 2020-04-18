#!/usr/bin/env python3

import argparse

def get_args():
    parser=argparse.ArgumentParser(
    description="""this script is used to get downsample file from sph file""")
    parser.add_argument('wav_scp', type=str,
                       help="""input wav.scp file, it has modified , but it is error file,
                            I will correct.""")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.wav_scp, 'r') as f:
        for line in f:
            line = line.strip('\n').split('|')
            line_sub = line[0].split(' ')
            print (line_sub[0] + " sph2pipe " + ' '.join(line_sub[2:]) + "| /usr/bin/sox -r 8000 - -c 1 -b 16 -t wav - |")


if __name__ == "__main__":
    main()
            

# how to do this script?
# current path:/home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data
# source-md/egs/project_2019_11/8k/english/add_data/downsample_sph_file_for_SWB1.py test/SWB1_wav_5.scp  
