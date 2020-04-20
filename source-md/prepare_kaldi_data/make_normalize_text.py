#!/usr/bin/env python3

# 2020 Ma Duo

import argparse
import re


def get_parse():
    parser = argparse.ArgumentParser(description="normalize text. e.g: delete '-'")
    parser.add_argument("text", type=str, help="transcript for wave data")
    args = praser.parse_args()
    return args


def main():
    args = get_parse()
    with open(args.text, "r", encoding="utf-8") as f:
        content = f.readlines()
    for line in content:
        line = line.strip().split(" ")
        utt = re.sub(r"-", r"", line[1:])
        print("{0} {1}".format(line[0], utt))


if __name__ == "__main__":
    main()
