#!/usr/bin/env python3

# 2020 Ma duo

# from utt2dur get segments file

import argparse

def get_parse():
    parser = argparse.ArgumentParser(
     description='from utt2dur get segments file, note here segments id is equal to wav id'
                 'first field is segments id , second field is wav id')
    parser.add_argument('utt2dur_file', type=str,
                       help='input utt2dur file path')
    args = parser.parse_args()
    return args

def main():
    args = get_parse()
    with open(args.utt2dur_file, 'r', encoding='utf-8') as f:
        utt2dur_file = f.readlines()
    for line in  utt2dur_file:
        line = line.strip().split(' ')
        print('{0} {1} {2} {3}'.format(line[0],line[0],0,line[1]))

if __name__ == '__main__':
    main()
