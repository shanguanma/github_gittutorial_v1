#!/usr/bin/env python3

import argparse
def get_parser():
    parser = argparse.ArgumentParser(
        description='convert a json to a transcription file with a token dictionary')
    parser.add_argument('result', type=str, help='json files')
    return parser
def main():
    parser = get_parser()
    args = parser.parse_args()
    with open(args.result, 'r') as f:
        context = f.readlines()
    for line in context:
        line = line.strip().split(" ")
        print(line[0]+".wav"," ".join(line[1:]))


if __name__ == "__main__":
    main()
