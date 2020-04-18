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
            if line[0].endswith('_A'):  
                print(line[0]+ ' sph2pipe -f wav -p -c 1 ' + line[8] + '|' + ' /usr/bin/sox -r 8000 - -c 1 -b 16 -t wav - |') 
            else:
                # line[0].endswith('_B')
                print(line[0]+ ' sph2pipe -f wav -p -c 2 ' + line[8] + '|' + ' /usr/bin/sox -r 8000 - -c 1 -b 16 -t wav - |') 
 

if __name__ == '__main__':
    main()

