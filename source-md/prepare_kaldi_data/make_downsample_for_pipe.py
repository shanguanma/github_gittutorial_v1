#!/usr/bin/env python3

# 2020 Ma Duo

import argparse

def get_parse():
    parser = argparse.ArgumentParser(
     description='make downsample wav.scp, e.g: from 16k-->8k, 22.5k -->8k')
    parser.add_argument('wav_scp', type=str,
                       help='input wav.scp file path, it may be 16k')
    args = parser.parse_args()
    return args
def main():
    args = get_parse()
    with open(args.wav_scp, 'r', encoding='utf-8') as f:
        wav_scp = f.readlines()
    for line in wav_scp:
        line = line.strip().split(' ')
        if line[-1] == '|' :  
           print(line[0]+' '+ '/usr/bin/sox -t wav'+ ' ' + line[2] + ' ' + '-r 8000 -c 1 -b 16 -t wav - |') 
        else:
            path = line[1].split('/')
            suffix = path[-1].split('.')[-1] 
            if suffix == 'wav':
                # wav format
                # example: /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/lexicon_comparison/run_imda_1a/data/train_imda_part3_ivr/wav.scp 
                print('{0} /usr/bin/sox -t wav {1} -r 8000 -c 1 -b 16 -t wav - |'.format(line[0], line[1]))
            else:
                # suffix == 'pcm':
                # pcm format
                print('{0} /usr/bin/sox  {1} -r 8000 -c 1 -b 16 -t wav - |'. format(line[0], line[1]))
 

if __name__ == '__main__':
    main()

