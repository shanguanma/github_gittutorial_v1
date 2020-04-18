#!/usr/bin/env python3
# source-md/egs/dev_work/ctc_prepared_data/prep_ctc_trans.py
 
# This python script converts the word-based transcripts into label sequences. The labels are
# represented by their indices. 

# refrence:eesen/asr_egs/wsj/utils/prep_ctc_trans.py


import sys

import argparse


if __name__ == "__main__":
    if len(sys.argv) < 4 or len(sys.argv) > 5:
        print ("Usage {0} <lexicon_file> <trans_file> <unk_word> [space_word]".format(sys.argv[0]))
        print ("current path : /home4/md510/w2019a/espnet-recipe/dev_work")
        print ("e.g: prepared phone unit :source-md/egs/dev_work/ctc_prepared_data/prep_ctc_trans.py\
                                          ctc_prepared_data/wsj_dict_phn/lexicon_numbers.txt \
                                          ctc_prepared_data/data_wsj/train_si84/text '<UNK>'")
        print("e.g: prepared char unit :source-md/egs/dev_work/ctc_prepared_data/prep_ctc_trans.py\
                                        ctc_prepared_data/wsj_dict_phn/lexicon_numbers.txt \
                                        ctc_prepared_data/data_wsj/train_si84/text '<UNK>' '<SPACE>'")
        print( "<lexicon_file> - the lexicon file in which entries have been represented by indices")
        print( "<trans_file>   - the word-based transcript file")
        print( "<unk_word>     - the word which represents OOVs in transcripts")
        print( "[space_word]   - optional, the word representing spaces in the transcripts")
        exit(1)
#    parser = argparse.ArgumentParser(
#      description="""This python script converts the word-based transcripts into label sequences. The labels are
#                   represented by their indices."""
#                   "current path : /home4/md510/w2019a/espnet-recipe/dev_work"
#                   " e.g: prepared phone unit :source-md/egs/dev_work/ctc_prepared_data/prep_ctc_trans.py"
#                   " ctc_prepared_data/wsj_dict_phn/lexicon_numbers.txt ctc_prepared_data/data_wsj/train_si84/text --unk-word "<UNK>" "
#                   " e.g: prepared char unit :source-md/egs/dev_work/ctc_prepared_data/prep_ctc_trans.py"
#                   " ctc_prepared_data/wsj_dict_phn/lexicon_numbers.txt ctc_prepared_data/data_wsj/train_si84/text --unk-word "<UNK>" "<SPACE>""
#                   "<lexicon_file> - the lexicon file in which entries have been represented by indices"
#                   "<trans_file>   - the word-based transcript file"
#                   "<unk_word>     - the word which represents OOVs in transcripts"
#                   "[space_word]   - optional, the word representing spaces in the transcripts")
#    parser.add_argument('lexicon_file',type=str,help="the lexicon file in which entries have been represented by indices")
#    parser.add_argument('trans_file',type=str,help="the word-based transcript file")
#    parser.add_argument('--unk-word',type=str, default="<UNK>",help="the word which represents OOVs in transcripts")
#    parser.add_argument('--space-word',type=str,default=None,choices=[None, "<SPACE>"],
#                        help="in the char case, you need to choice SPACE,in the phone case, you need to choice None ")
#    args = parser.parse_args()
    dict_file = sys.argv[1]
    trans_file = sys.argv[2]
    unk_word = sys.argv[3]
    is_char = False
    if len(sys.argv) == 5:
        is_char = True
        space_word = sys.argv[4]
    # read lexicon into a dictionary data structor
    with open(dict_file,'r') as f:
        dict = {}
        context = f.readlines()
    for line in context:
        line = line.strip('\n')
        line_split = line.split(' ') # default is also space.
        word = line_split[0]
        letter = ''
        for n in range(1, len(line_split)):
            letter += line_split[n] + ' '
        dict[word] = letter.strip('\n') # remove line break
    
    #print (dict)
    # assume that per line is formated as 'uttid word1 word2 word3 ... 'with no multiple space.
    with open(trans_file,'r') as fh:
        out_line = ''
        context_trans = fh.readlines()
    for line in context_trans:
        line = line.strip('\n') # line should be strings
        #print(type(line))
        
        while '  ' in line:
            line = line.replace( '  ',' ') # remove multiple space.
        
        uttid = line.split(' ')[0] # the first field is always uttid.
        #print("line.split(' ') is :" ,type(line.split(' ')))
        trans = line.replace(uttid , '').strip() # trans should be strings

        if is_char:
            trans = trans.replace(' ', ' ' + space_word + ' ' ).strip()
        splits = trans.split(' ')
        
        #out_line += uttid + ' '
        for n in range(len(splits)):
            try:  
                out_line += dict[splits[n]] + ' '
            except Exception:
                out_line += dict[args.unk_word] + ' '
        #print(out_line.strip())
        print("{0} {1}".format(uttid, out_line))


# test 

# source-md/egs/dev_work/ctc_prepared_data/prep_ctc_trans.py ctc_prepared_data/wsj_dict_phn/lexicon_numbers.txt text_10 "<UNK>"  
