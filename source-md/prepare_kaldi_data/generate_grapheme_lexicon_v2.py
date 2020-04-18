#!/usr/bin/env python3

# maowang@ntu,20191219
# Ma Duo 2020-3-11
#        2020-4-8
import argparse
import os 
from pathlib import Path
from typing import Union
from typing import Iterable

def get_parse():
    parser = argparse.ArgumentParser(
     description="make grapheme lexicon.txt")
    parser.add_argument('--wordlist_en', type=str, required=True,
                        help='it may be single column text file which word in per line or two column text file'
                             'which is first column is word and'
                             ' the second column is the number of occurrences of the corresponding word')
    parser.add_argument('--dict_dir',type=str, required=True,
                        help='it is path of dictionary. it need to be created manually')
    args = parser.parse_args()
    return args

def BuildGraphemeLexicon(wordlist_file: Iterable[Union[str, Path]], dict_dir: Union[str, Path]):
    dict_dir = Path(dict_dir)
    with open(wordlist_file, 'r', encoding='utf-8') as f,\
        open(dict_dir / f"pre_lexicon.txt",'w', encoding='utf-8') as fo:
        for line in f:
            line = line.strip().split(' ')
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
                    if i == 0 :
                        pronunciation = word[i] + '_WB' + ' '
                    elif i != 0 and i != len(word) -1 :
                        pronunciation += word[i] + ' '
                    else:
                        pronunciation += word[i] + '_WB'
            fo.write(word + ' ' + pronunciation + '\n')

## make extra_questions.txt, it can be empty.
def creat_extra_questions(dict_dir: Union[str, Path]):
    dict_dir = Path(dict_dir)
    if not os.path.exists(dict_dir / f"extra_questions.txt"):
        f = open(dict_dir / f"extra_questions.txt", 'w') 
        f.close()

# make optional_silence.txt , it only contain "SIL" .
def creat_optional_silence(dict_dir: Union[str, Path]):
    dict_dir = Path(dict_dir)
    if not os.path.exists(dict_dir / f"optional_silence.txt"):
        f = open(dict_dir / f"optional_silence.txt", 'w')
        f.write("SIL\n")
        f.close()

# make  silence_phones.txt it contain three symbols (e.g: SIL,<sss>, <oov>), per line has a symbol    
def creat_silence_phones(dict_dir: Union[str, Path]):
    dict_dir = Path(dict_dir)
    if not os.path.exists(dict_dir / f"silence_phones.txt"):
        f = open(dict_dir / f"silence_phones.txt", 'w')
        f.write("SIL\n<sss>\n<oov>\n")
        f.close()

# make nonsilence_phones.txt 
def creat_nonsilence_phones(dict_dir: Union[str, Path]):
    dict_dir = Path(dict_dir)
    d =list()
    pre_dict = dict_dir / f"pre_lexicon.txt"
    with open(pre_dict, 'r', encoding="utf-8")as fi:
        for line in fi:
            line = line.strip().split(" ")
            for i in line[1:]:
                if i not in d:
                   d.append(i)
        d = [ i for i in d if not i.startswith('#')]
        with open(dict_dir / f"nonsilence_phones.txt", 'w', encoding="utf-8") as fo:
            for char in sorted(d):
                fo.write(char + "\n")

# append two pairs  for lexicon.txt(e.g: pre_dict)
#  <noise> <sss>
#  <unk> <oov> 
def creat_lexicon( dict_dir: Union[str, Path]):
    dict_dir = Path(dict_dir)
    lex_dict = dict()
    pre_dict = dict_dir / f"pre_lexicon.txt"
    with open(pre_dict, 'r', encoding="utf-8") as fi:
        for line in fi:
            line = line.strip().split(' ')
            lex_dict[line[0]] = ' '.join(line[1:])
     
        lex_dict["<noise>"] = "<sss>"
        lex_dict["<unk>"] = "<oov>"
        # Filter out words starting with #
        lex_dict = {key: value for key, value in lex_dict.items() if not key.startswith('#')}
        lexicon_list = sorted(lex_dict.items(), key=lambda item: item[0])
    with open(dict_dir / f"lexicon.txt", 'w', encoding="utf-8")as fo:
        for i  in lexicon_list:
            fo.write(' '.join(i)+ '\n')
        
def main():
    args = get_parse()
    BuildGraphemeLexicon(args.wordlist_en, args.dict_dir)
    creat_extra_questions(args.dict_dir)
    creat_optional_silence(args.dict_dir)
    creat_silence_phones(args.dict_dir)
    creat_nonsilence_phones(args.dict_dir)
    creat_lexicon(args.dict_dir)
     
if __name__=='__main__':
    main()

# how to run this script?
# current path: /home4/md510/w2020/kaldi-recipe/egs/malay_cts
# source-md/prepare_kaldi_data/generate_grapheme_lexicon_v2.py --wordlist_en test/wordlist_10.txt  --dict_dir test/dict_sub

"""
 [md510@node08 malay_cts_new]$ head kaldi_data/16k/dict_16k/extra_questions.txt 
 [md510@node08 malay_cts_new]$ head kaldi_data/16k/dict_16k/lexicon.txt 
<noise> <sss>
<unk> <oov>
a a_WB
aa a_WB a_WB
aaa a_WB a a_WB
aab a_WB a b_WB
aadk a_WB a d k_WB
aalco a_WB a l c o_WB
aam a_WB a m_WB
aamir a_WB a m i r_WB
[md510@node08 malay_cts_new]$ head kaldi_data/16k/dict_16k/nonsilence_phones.txt 
'
-
0_WB
a
a_WB
b
b_WB
c
c_WB
d
[md510@node08 malay_cts_new]$ head kaldi_data/16k/dict_16k/optional_silence.txt 
SIL
[md510@node08 malay_cts_new]$ head kaldi_data/16k/dict_16k/silence_phones.txt 
SIL
<sss>
<oov>
"""
