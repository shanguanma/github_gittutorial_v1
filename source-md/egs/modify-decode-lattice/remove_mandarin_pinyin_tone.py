#!/usr/bin/env python3

# Copyright 2019 Ma Duo
# Apache 2.0

# Indentation is two spaces

import argparse
import sys
sys.path.append('steps/libs')
import common as common_lib

def get_args():
  parser = argparse.ArgumentParser(
   description=""" This script converts a syllable dict to remove
   tone of pinyin ,then get new sylable dict. its input required
   have three column, the first column is a complete pinyin with 
   tone,the second column is the initial corresponding to pinyin.
   the third column is the vowel corresponding to pinyin. the format 
   is as follows:
   <a complete pinyin with tone> <initial> <vowel with tone>
   for example:
   a1 ga a1
   a2 ga a2
   a3 ga a3
   a4 ga a4
   a5 ga a5
   ai1 ga ai1
   ai2 ga ai2
   ai3 ga ai3
   ai4 ga ai4
   an1 ga an1
   its output format is as follows:
   <a complete pinyin without tone> <initial> <vowel with tone> 
   a ga_man a1_man 
   a ga_man a2_man 
   a ga_man a3_man 
   a ga_man a4_man 
   a ga_man a5_man""")
  parser.add_argument("pre_syllable_dict",type=str,
                      help="Input syllable dictionary its pinyin with tone")
  #parser.add_argument("dict_pinyin_without_tone",type=str,
  #                    help="Output syllable dictonary,its pinyin without tone")
  args = parser.parse_args()
  return args

def main():
  args = get_args()
  with common_lib.smart_open(args.pre_syllable_dict) as pre_syllable_dict_file:
    for line in pre_syllable_dict_file:
      if len(line.strip().split())==2:
        pinyin, initial_id = line.strip().split()
        #old_pinyin_dict[pinyin]=[initial_id,vowel_id]
        print("{0}\t{1}".format(pinyin[:-1],initial_id+"_man"))
      else:
        pinyin, initial_id,vowel_id = line.strip().split()
        print("{0}\t{1} {2}".format(pinyin[:-1],initial_id+"_man",vowel_id+"_man"))
if __name__ == '__main__':
  main() 
      
    

#test 
# source-md/egs/modify-decode-lattice/remove_mandarin_pinyin_tone.py mandarin_dict_from_haihua/if-syl-dict-update_v1.txt > mandarin_dict_from_haihua/if-syl-dict-update_v1_remove_tone.txt 





