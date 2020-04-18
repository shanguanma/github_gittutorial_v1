#!/usr/bin/env python3


# copyright 2019 Ma Duo
# Apache 2.0

import argparse
import codecs
import logging

def get_args():
    parser = argparse.ArgumentParser(
     description=""" Compare two dictionaries to get 
        the different wordlist.""")
    parser.add_argument("input_dict_1", type=str,
                        help="input file, it is the big dict path")
    parser.add_argument("input_dict_2",type=str,
                        help="input file, it is the small dict path")
    args = parser.parse_args()
    return args

def main():
    # 1. get wordlist(unique) from input_dict_1 
    #    get wordlist(unique from input_dict_2
    # 2. get difference from two wordlists  
    args = get_args()
    with open(args.input_dict_1, 'r', encoding="utf-8") as f:
        dictionary_1 = f.readlines()
    with open(args.input_dict_2, 'r', encoding="utf-8") as f:
        dictionary_2 = f.readlines() 
    # remove repeat element and sort it
    dict_1_words = [line_1.strip().split()[0] for line_1 in dictionary_1]
    #print(dict_1_words)
    dict_2_words = [line_2.strip().split()[0] for line_2 in dictionary_2]
    #print(dict_2_words)

    difference_with_pronunciation = [i for i in dict_1_words if i not in  dict_2_words] 
    print("All difference prounciation are :",difference_with_pronunciation)
    difference = list(set(dict_1_words).difference(set(dict_2_words)))
    print(difference)
    print("The difference wordlist length of two dictionaries is:",len(difference))
    print("Two dictionaries have different prounciation")
    for line_1 in dictionary_1:
        line_1 = line_1.strip().split()
        if line_1[0] in difference_with_pronunciation:
            print("{0}\t{1}".format(line_1[0]," ".join(line_1[1:]))) 
if __name__ == "__main__":
    main() 



# how to run?
# source-md/egs/project_2019_11/8k/english/lexicon_comparison/get_different_wordlist_for_dicts.py run_ntu_inhouse_1a/data/dict_ntu_inhouse/lexicon.txt  run_ntu_inhouse_1b/data/dict_ntu_inhouse_without_mandarin/lexicon.txt

# source-md/egs/project_2019_11/8k/english/lexicon_comparison/get_different_wordlist_for_dicts.py run_imda_1a/data/dict_imda/lexicon.txt  run_ntu_inhouse_1b/data/dict_ntu_inhouse_without_mandarin/lexicon.txt 
