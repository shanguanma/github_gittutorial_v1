#!/usr/bin/env python3

# Ma Duo

import argparse

def get_parse():
    args = argparse.ArgumentParser(
     description='rewrite wave path in wav.scp.')
    parser.add_argument('wav_scp', type=str,
                       help='input wav.scp file path')
    parse.add_argument("path", type=str,help='to replace path part')
    args = parser.parse_args()
    return args

def main():
    args = get_parse()
    with open(args.wav_scp,'r',encoding='utf-8') as f:
        wav_scp = f.readlines()
    for line in wav_scp:
         
