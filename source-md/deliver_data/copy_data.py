#!/usr/bin/env python3

# 2020 NTU Ma Duo

import argparse
import os
import shutil 
def get_parse():
    parser = argparse.ArgumentParser(
     description='make copy wave file from  wav.scp')
    parser.add_argument('wav_scp', type=str,
                       help='input wav.scp file path')
    parser.add_argument('path_prefix', type=str,
                       help='you will store wave file prefix path')
    args = parser.parse_args()
    return args
def main():
    args = get_parse()
    #path_prefix = '/home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/test/read_data'
    path_prefix = args.path_prefix
    with open(args.wav_scp, 'r', encoding='utf-8') as f:
        wav_scp = f.readlines()
    for line in wav_scp:
        line = line.strip().split()
        line_path = line[1].split('/')
        line_path_common = '/'.join(line_path[4:-1])
        # print(os.path.dirname(line[1]))
        path=path_prefix+'/'+line_path_common
        if not os.path.exists(path):
            os.makedirs(path)
        shutil.copy(line[1], path)
        print('copy finish! ')

   
if __name__ == '__main__':
   main()


# test 
# current path:
# /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data
# source-md/deliver_data/copy_data.py  deliver_wang/ntu-data-dev/formatted/read/wav.scp  /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/deliver_wang/ntu-data-dev/data/read
