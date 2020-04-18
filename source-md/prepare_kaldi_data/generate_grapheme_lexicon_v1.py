#!/usr/bin/env python3

# maowang@ntu,20191219
# Ma Duo 2020-3-11
import argparse
def get_parse():
    parser = argparse.ArgumentParser(
     description="make grapheme lexicon.txt")
    parser.add_argument('--wordlist_en', type=str, required=True,
                        help='it may be single row text file which word in per line or two row text file'
                             'which is first row is word and'
                             ' the second column is the number of occurrences of the corresponding word')
    args = parser.parse_args()
    return args

def BuildGraphemeLexicon(wordlist_file):
    with open(wordlist_file, 'r', encoding='utf-8') as f:
        content = f.readlines()
    for line in content:
        line = line.strip().split('\t')
        if len(line) == 2:
            word = ''.join(line[0])
            #print(type(word))
            #print(word)
        elif len(line) == 1:
            word = ' '.join(line)
            #print(type(word))
        else:
            raise RuntimeError(f"Not supported: {len(line)}, only supported single row or two rows text file")
        if len(word) == 1:
            pronunciation = word + '_WB'
        else:
            for i in range(len(word)):
                if i == 0:
                    pronunciation = word[i] + '_WB' + ' '
                elif i != 0 and i != len(word) -1:
                    pronunciation += word[i] + ' '
                else:
                    pronunciation += word[i] + '_WB'
        print(word + ' ' + pronunciation)


def main():
    args = get_parse()
    BuildGraphemeLexicon(args.wordlist_en)

if __name__=='__main__':
    main()

# how to run this script?
# current path:/home4/md510/w2020/kaldi-recipe/egs/maison2_nana
# source-md/prepare_kaldi_data/generate_grapheme_lexicon_v1.py  --wordlist_en test/oov_count_10.txt > test/oov_lexicon_10.txt 
