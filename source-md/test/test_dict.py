#!/usr/bin/env python3

import argparse
#import sys
#sys.path.append('steps/libs')
#import common as common_lib
# how to run the script?
# current folder path:/home3/md510/w2019a/espnet-recipe/dev_work
# python3 source-md/test/test_dict.py data/lang_1char/train_sub_per_spk_units.txt 
def get_args():
  parser = argparse.ArgumentParser(
   description=""" This script display e2e dictionary""")
  parser.add_argument("dict",type=str,
                       help="Input dict path")
  args = parser.parse_args()
  return args


def main():
  args = get_args()
  # load dictionary
  with open(args.dict, 'rb') as f:
    # because dict file contains two columns
    # first column is letter,second column is corresponding id of letter.
    # note: because I use 'rb' mode ,so it display utf-8 character.
    # f.readlines() is list type, # per line store a element of list.
    dictionary = f.readlines()
    
    print ("dictionary:",dictionary)
  print ("type of dictionary: ", type(dictionary))
  # char_list is list ,but its per element is only letter, not id of letter.
  char_list = [entry.decode('utf-8').split(' ')[0] for entry in dictionary]
  print ("char_list:",char_list)
  char_list.insert(0, '<blank>')
  char_list.append('<eos>')
  # I convert char_list to char_list dict
  char_list_dict = {x: i for i, x in enumerate(char_list)}
  print ("add <blank> and <eos> ,char_list_dict:", char_list_dict)
  n_vocab = len(char_list) 
  print("n_vocab, it is n_vocab of embedding  and it also is units of last output layer:",n_vocab)


if __name__ == "__main__":
  main()
