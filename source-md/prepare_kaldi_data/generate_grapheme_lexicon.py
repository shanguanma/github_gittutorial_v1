#!/usr/bin/env python3

# maowang@ntu,20191219

import sys

if len(sys.argv) != 2:
    print("python3 generate_grapheme_lexicon.py wordlist_en", file=sys.stderr)
    exit(1)
else:
    wordlist_file = sys.argv[1]


def BuildGraphemeLexicon(wordlist_file):
    for word in open(wordlist_file):
        word = word.strip()
        if len(word) == 1:
            pronunciation = word + "_WB"
        else:
            for i in range(len(word)):
                if i == 0:
                    pronunciation = word[i] + "_WB" + " "
                elif i != 0 and i != len(word) - 1:
                    pronunciation += word[i] + " "
                else:
                    pronunciation += word[i] + "_WB"
        print(word + "\t" + pronunciation)


def main():
    BuildGraphemeLexicon(wordlist_file)


if __name__ == "__main__":
    main()
