#!/usr/bin/env python3

# copyright 2019 Ma Duo

import argparse

def get_args():
    parser = argparse.ArgumentParser( 
     description=""" Function of this script is used to replace keywords with real keywords
                 in keywords search result.
                 you can see it in source-md/egs/modify-decode-lattice/run_v3.sh""") 
    parser.add_argument("keyword_search_result",type=str,
                        help="keyword script output file,then I covert frame to time."
                              " for example: /home4/md510/w2019a/kaldi-recipe/chng_project_2019/"
                              "exp/keyword_tung_v3/search_results/dev/result_total_time_type")

    parser.add_argument("keywords_text", type=str,
                         help="keywords list text. its content is as follow:"
                         " <keyword-id> <keyword>"
                         " kw01 xu hai hua "
                         " kw02 ho thi nga ")
    
    args = parser.parse_args()
    return args

def get_keywords(keywords_text_filename):
    id2keywords = {}
    with open(keywords_text_filename,'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        id2keywords[line_split[0]] = ' '.join(line_split[1:])
    return id2keywords


#def get_keyword_search_result(keyword_search_result_filename):
#    id2keyword_result = {}
#    with open(keyword_search_result_filename,'r')  as fh:
#        content = fh.readlines()
#    for line in content: 
#        line = line.strip('\n')
#        line_split = line.split()
#        id2keyword_result[line_split[0]] = ' '.join(line_split[1:])
#    return id2keyword_result


def main():
    args = get_args()
    id2keywords = get_keywords(args.keywords_text)
    with open(args.keyword_search_result) as f:
        content = f.readlines()
    for line in content:
        line = line.strip('\n')
        line_split = line.split()
        if line_split[0] in id2keywords:
            new_line = ' '.join(line_split[1:]) + ' ' + id2keywords[line_split[0]]     
            
            #print("{0} {1}".format(line_split[0], new_line))   
            print("{0}".format(new_line))
        
            
if __name__ == "__main__":
    main()



# test
# current path:/home4/md510/w2019a/kaldi-recipe/chng_project_2019
# source-md/egs/modify-decode-lattice/make_replace_keywords_in_search_result.py exp/keyword_tung_v3/search_results/dev/result_total_time_type  /home2/tungpham/hotlist_rescoring_Maduo_v3/script_index_search/keywords.txt > exp/keyword_tung_v3/search_results/dev/keywords_align




