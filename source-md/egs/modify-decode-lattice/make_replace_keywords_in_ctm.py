#!/usr/bin/env python3

# copyright 2019 Ma Duo

import argparse
#from  collections import defaultdict
def get_args():
    parser = argparse.ArgumentParser(
     description=""" Function of this script is used to replace keywords with real keywords
                 in word boundary ctm file.
                 you can see it in source-md/egs/modify-decode-lattice/run_v3.sh""")
    parser.add_argument("word_boundary_ctm",type=str,
                        help=" It is a ctm file, it word-level align file, it include word "
                             " boundary. In this example,This file I am using comes from the " 
                             " one best path ctm file in one pass decoder. "
                             " Its format is as follows:"
                             " <utt-id> <channel> <start-time> <duration-time> <word>"
                             " audio1-1 1 6.72 0.27 ling "
                             " audio1-1 1 6.99 0.21 zhi "
                             " audio1-1 1 7.20 0.45 hao "
                             " audio1-1 1 8.10 0.27 hao "
                             " audio1-1 1 8.37 0.30 jing "
                             " audio1-1 1 8.67 0.45 tune "
                             " audio1-1 1 9.48 0.81 liberty " 
                             " audio1-1 1 11.04 0.27 how "
                             " audio1-1 1 11.31 0.24 dan "
                             " audio1-1 1 11.55 0.45 yuan " )
    parser.add_argument( "keywords_align", type=str,
                        help=" it is from output of this script "
                             " (e.g:source-md/egs/modify-decode-lattice "
                             " /make_replace_keywords_in_search_result.py) "
                             " its format is as follows: "
                             " <utt-id> <start-time> <end-time> <keywords-name> "
                             " audio1-1 6.72 7.65 lim zhi hao "
                             " audio1-1 11.04 12 ho dan yuan " 
                             " for example:home4/md510/w2019a/kaldi-recipe/chng_project_2019/ "
                             " exp/keyword_tung_v3/search_results/dev/keywords_align ")

    parser.add_argument("output_modify_transcript", type=str,
                        help=" using keyword boundary information to modify one pass ctm file, get"
                             " modified file")
    args  = parser.parse_args()
    return args
#  exp/keyword_tung_v3/search_results/dev/keywords_align
#  asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/score_ctm/score_10_1.0/test_wavdata_v3.utt_ctm
# d = {} # 一个普通的字典
# d.setdefault('a', []).append(1)
# d.setdefault('a', []).append(2)
# d.setdefault('b', []).append(4)
# print (d)
# {'a': [1, 2], 'b': [4]} 
def get_keywords_align(keywords_align_filename):
    id2keywords_align = {}
    with open(keywords_align_filename, 'r') as f:
        context = f.readlines()
    for line in context:
        line = line.strip("\n")
        line_split = line.split()
        key = line_split[0]
        #value = [tuple((float(line_split[1]),float(line_split[2]))),tuple(line_split[3:])]
        value = list((float(line_split[1]),float(line_split[2]),line_split[3:]))
        id2keywords_align.setdefault(key,[]).append(value)
 
    return id2keywords_align


def get_word_boundary_ctm(word_boundary_ctm_filename):
    id2word_boundary_ctm = {}
    with open(word_boundary_ctm_filename, 'r') as f:
        context =f.readlines()
    for line in context:
        line = line.strip("\n")
        line_split = line.split()
        key = line_split[0]
        value = list((float(line_split[2]),float(line_split[3]),line_split[4]))
        id2word_boundary_ctm.setdefault(key,[]).append(value)
    return id2word_boundary_ctm

if __name__ == '__main__':
    args = get_args()
    
    id2word_boundary_ctm = get_word_boundary_ctm(args.word_boundary_ctm)
    id2keywords_align = get_keywords_align(args.keywords_align) 
    #print(id2word_boundary_ctm)
    #print(id2keywords_align)
                        
    for key, value in id2keywords_align.items():
        if key in id2word_boundary_ctm:
            for i in range(len(value)):
                for y in range(len(id2word_boundary_ctm[key])):
                    if value[i][0] == id2word_boundary_ctm[key][y][0]:
                        if len(value[i][2]) == 3 and \
                        value[i][1]>=id2word_boundary_ctm[key][y][0]+id2word_boundary_ctm[key][y][1]+id2word_boundary_ctm[key][y+1][1]+id2word_boundary_ctm[key][y+2][1]:
                            id2word_boundary_ctm[key][y][2] = value[i][2]
                            id2word_boundary_ctm[key][y+1][2] = ''
                            id2word_boundary_ctm[key][y+2][2] = ''
                           
                        elif len(value[i][2]) == 2 and \
                        value[i][1]>=id2word_boundary_ctm[key][y][0]+id2word_boundary_ctm[key][y][1]+id2word_boundary_ctm[key][y+1][1]:
                            id2word_boundary_ctm[key][y][2] = value[i][2]
                         
                            id2word_boundary_ctm[key][y+1][2] = ''
                            
                        else:
                            len(value[i][2]) == 1 and \
                            value[i][1]>=id2word_boundary_ctm[key][y][0]+id2word_boundary_ctm[key][y][1]
                            id2word_boundary_ctm[key][y][2] = value[i][2]
                       
    #for key , value in id2word_boundary_ctm.items():
    #for key in id2word_boundary_ctm.keys():
    #    for value in id2word_boundary_ctm.value

    #for key, value in id2word_boundary_ctm.items():
    #    for i in range(len(value)):
    #        if isinstance(value[i][2],list):
    #            print (key, ' '.join(value[i][2]))
    #        else:
    #            print (key, value[i][2])
    with open(args.output_modify_transcript,'w') as writer:
        for key,value in id2word_boundary_ctm.items():
            str_list = [' '.join(value[i][2]) if isinstance(value[i][2],list) else value[i][2] for i in range(len(value))]
            # Remove empty strings from a list of strings
            # str_list = list(filter(None, str_list)) # fastest
            print("{0} {1}".format(key, ' '. join(list(filter(None,str_list)))),file=writer)                                


# test
# 
# source-md/egs/modify-decode-lattice/make_replace_keywords_in_ctm.py  asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/score_ctm/score_10_1.0/test_wavdata_v3.utt_ctm exp/keyword_tung_v3/search_results/dev/keywords_align  asr_models_english_zhiping/chain1024tdnnf/decode_test_wavdata_v3/words_boundary_modified_one_pass/modified_one_pass_transcript
