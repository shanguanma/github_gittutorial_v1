#!/usr/bin/env python3


# copyright 2019 Ma Duo
# Apache 2.0

import argparse
import codecs
import logging

def get_args():
    parser = argparse.ArgumentParser(
     description=""" Compare two dictionaries to get 
       a sub-dictionary of the same wordlist.""")
    parser.add_argument("input_dict_1", type=str,
                        help="input file, it is the first dict path")
    parser.add_argument("input_dict_2",type=str,
                        help="input file, it is the second dict path")
    parser.add_argument("output_dict_1", type=str,
                       help="corresponding to sub dict of input_dict_1")
    parser.add_argument("output_dict_2", type=str,
                       help="corresponding to sub dict of input_dict_2")
    args = parser.parse_args()
    return args

def main():
    # 1. get wordlist(unique) from input_dict_1 
    #    get wordlist(unique from input_dict_2
    # 2. get intersection from two wordlist  
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

    #intersection = [i for i in dict_1_words if i in dict_2_words] 
    intersection = list(set(dict_1_words).intersection(set(dict_2_words)))
    print(intersection)
    print("The common wordlist length of two dictionaries is:",len(intersection))
    output_dict_1 = codecs.open(args.output_dict_1, 'w', encoding="utf-8") 
    output_dict_2 = codecs.open(args.output_dict_2, 'w', encoding="utf-8")
    for line_1 in dictionary_1:
        line_1 = line_1.strip().split()
        if line_1[0] in intersection:
             output_dict_1.write("{0}\t{1}\n".format(line_1[0], " ".join(line_1[1:])))
    #output_dict_1.close()
    for line_2 in dictionary_2:
        line_2 = line_2.strip().split()
        if line_2[0] in intersection:
             output_dict_2.write("{0}\t{1}\n".format(line_2[0], " ".join(line_2[1:])))
    output_dict_1.close()
    output_dict_2.close() 
if __name__ == "__main__":
    main() 



# how to run?
# source-md/egs/project_2019_11/8k/english/lexicon_comparison/get_same_wordlist_for_dicts.py  test/ntu_inhouse_1b_10.txt test/imda_lexicon_10.txt  test/ntu_inhouse_1b_10_sub.txt  test/imda_lexicon_10_sub.txt 
