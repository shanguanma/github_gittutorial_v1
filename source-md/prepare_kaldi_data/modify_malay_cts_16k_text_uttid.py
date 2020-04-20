#!/usr/bin/env python3


import sys

if len(sys.argv) != 2:
    print("python3 generate_grapheme_lexicon.py text_file", file=sys.stderr)
    exit(1)
else:
    text_file = sys.argv[1]


def modify_uttid(text_file):
    with open(text_file) as f:
        for line in f:
            line = line.strip().split(" ")
            line_sub = line[0].split("-")
            line_sub_su = line_sub[2].split("_")

            print(
                line_sub[0]
                + "-"
                + line_sub[1]
                + "-"
                + "_".join(line_sub_su[:4])
                + "_mic_"
                + line_sub_su[4]
                + " "
                + " ".join(line[1:])
            )


def main():
    modify_uttid(text_file)


if __name__ == "__main__":
    main()
