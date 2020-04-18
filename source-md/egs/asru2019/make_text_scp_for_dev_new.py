#!/usr/bin/env python3

import argparse
import sys
sys.path.append('steps/libs')
import common as common_lib
import os

def get_args():
    parser = argparse.ArgumentParser(
     description=""" the script makes kaldi text.scp from text path
     text path format is as follows:
     <text_path>
     for example:
     /home4/md510/w2019a/espnet-recipe/asru2019/data/ASRUn20/data/G1701/session01/T0426G1701_S01010043.txt
     text.scp format is as follows:
     <utt-id> <text_path>
     for example:
     G1701T0426G1701S01010043 /home4/md510/w2019a/espnet-recipe/asru2019/data/ASRUn20/data/G1701/session01/T0426G1701_S01010043.txt
     """)
    parser.add_argument("text_path_file",type=str,
                       help="input file")
    #parser.add_argument("kaldi_text_scp",type=str,
    #                   help="output file")
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    kaldi_text_scp = {}
    with open(args.text_path_file, 'r', encoding='utf-8') as freader:
        #common_lib.smart_open(args.kaldi_text_scp,'w') as fwriter:
        for line in freader:
            line = line.strip()
            #line_split = line.split('/')
            #print(line_split) 
           
            line_seperate_extention = line.split('.')
            line_split = line_seperate_extention[0].split('/') 
            line_fix = line_split[-1].split("_")
            print("{0} {1}".format(line_split[-3]+line_fix[0]+line_fix[1],line))
             



if __name__ == "__main__":
    main()

# test
# current path:/home4/md510/w2019a/espnet-recipe/asru2019
# source-md/egs/asru2019/make_text_scp_for_dev_new.py run_1b/data/dev_new/text_1_5 > run_1b/data/dev_new/text_1_5_scp 
