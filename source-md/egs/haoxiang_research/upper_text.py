#!/usr/bin/env python3

import argparse
import re
def get_args():
    parser=argparse.ArgumentParser(
     description= """covert lower english  utterance to upper text.""")
    parser.add_argument('text',type=str,help="input text file")
    args = parser.parse_args()
    return args



def main():
    args=get_args()
    with open (args.text,'r') as f:
       for line in f:
           line = line.strip('\n')
           line_list = line.split()
           #print(line_list[0],' '.join(line_list[1:]))
           str_= ' '.join(line_list[1:])
       
           print('{0} {1}'.format(str(line_list[0]),str_.upper()))


if __name__ == "__main__":
    main()

