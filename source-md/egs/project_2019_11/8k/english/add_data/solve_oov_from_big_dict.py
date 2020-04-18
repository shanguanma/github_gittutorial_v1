#!/usr/bin/env python3

import argparse
import re

def get_args():
    parser=argparse.ArgumentParser(
    description=""" this is a solution to oov,
    I select oov words from big dictionary, then add to small dictionary.""")
    parser.add_argument('oov_list', type=str, 
                        help="""input oov word list, the first column is word,
                        the second column is number of time""")
    parser.add_argument('biggest_dict', type=str, help='the big dictionary')
    args = parser.parse_args()
    return args

def main():
    args = get_args()
    oov_list=dict()
    with open(args.oov_list, 'r') as f:
        context = f.readlines()
    for line in context:
        line = line.strip('\n').split('\t')
        oov_list[line[0]] = line[1]
        #print("{0}\t{1}".format(line[0], line[1]))
        #oov_list.append(line[0])
    #print(len(oov_list))
    
    with open(args.biggest_dict, 'r') as f:
        for line in f:
            line_list = line.strip('\n').split('\t')
            if line_list[0] in oov_list.keys():
                print("{0}\t{1}".format(line_list[0], ' '.join(line_list[1:])))

if __name__ == "__main__":
   main()
        
        
    
     
    
    
