#!/usr/bin/env python3

# Copyright 2019 Ma Duo
import argparse
import codecs

def get_args():
    parser = argparse.ArgumentParser(
     description=""" Remove from the English dictionary 
     the same pronunciation of word as man_wordlist"""  )  

    parser.add_argument("english_dict",type=str,
                        help="input files, it is the object to be cleaned")
    parser.add_argument("man_wordlist", type=str,
                        help="""use pinyin to represent mandarin, 
                        per line contain a mandarin character """)
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    with open(args.english_dict, 'r', encoding='utf-8') as f:
        dictionary = f.readlines()
    with open(args.man_wordlist, 'r', encoding='utf-8') as f_1:
        wordlist = f_1.readlines()
    man_wordlist = [line.strip() for line in wordlist]
    for line in dictionary:
        line = line.strip().split()
        if line[0] not in man_wordlist:
            print("{0}\t{1}".format(line[0],' '.join(line[1:])))

        
if __name__ == "__main__":
    main()

# how to run it ?

