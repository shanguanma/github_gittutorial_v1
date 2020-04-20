#!/usr/bin/env python3

# Ma Duo 2020-3-12

import argparse
from typing import List
from typing import Dict
from pathlib import Path


def get_parse():
    parser = argparse.ArgumentParser(
        description="select part lexicon from big" "lexicon by specifying wordlist file"
    )
    parser.add_argument(
        "--trainset_en_text",
        type=str,
        required=True,
        help="kaldi format" "text file of train set," "it should be english word text",
    )
    parser.add_argument(
        "--big_lexicon",
        type=str,
        required=True,
        help="kaldi required pronunciton lexicon.txt" "file",
    )
    args = parser.parse_args()
    return args


def text2wordlist(trainset_en_text) -> List[str]:
    # this function is equal to shell script:
    # cut -f 2- -d " " trainset_en_text | tr " " "\n" | sort | uniq
    wordlist = []
    assert isinstance(trainset_en_text, (Path, str))
    trainset_en_text = Path(trainset_en_text)
    with trainset_en_text.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(" ")
            for word in line[1:]:
                if word not in wordlist:
                    wordlist.append(word)
        # print("wordlist : ",len(sorted(wordlist)))
        # print(sorted(wordlist))
        return sorted(wordlist)


def lexicon2dict(big_lexicon_file) -> Dict[str, str]:
    lexicon_dict = {}
    assert isinstance(big_lexicon_file, (Path, str))
    big_lexicon_file = Path(big_lexicon_file)
    with big_lexicon_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip().split(" ")
            lexicon_dict[line[0]] = " ".join(line[1:])
        # print(len(lexicon_dict))
        return lexicon_dict


def select_lexicon(trainset_en_text, big_lexicon_file) -> str:
    wordlist = text2wordlist(trainset_en_text)
    lexicon_dict = lexicon2dict(big_lexicon_file)
    # sub_lexicon = {key: value for key, value in lexicon_dict.item() if key in wordlist }
    for key, value in lexicon_dict.items():
        if key in wordlist:
            print("{0} {1}".format(key, value))


if __name__ == "__main__":
    args = get_parse()
    # text2wordlist(args.trainset_en_text)
    # lexicon2dict(args.big_lexicon)
    select_lexicon(args.trainset_en_text, args.big_lexicon)

# how to run this script?
# current path:/home4/md510/w2020/kaldi-recipe/egs/maison2_nana
# source-md/prepare_kaldi_data/select_lexicon_from_big_lexicon.py
#    --trainset_en_text test/text_10 \
#    --big_lexicon test/ubs_dictv0.3_merge/lexicon.txt
