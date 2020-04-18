#!/usr/bin/env python3

# copyright 2019 Ma Duo
# Apache 2.0
import argparse
import sys
sys.path.append('steps/libs')
import common as common_lib


def get_args():
    parser=argparse.ArgumentParser(
     description="""Function of this script is using postion_independent_phone_id instead of
      position_dependent_phone in ctm style file""")

    parser.add_argument("position_dependent_phone_mark", type=str,
                       help="Input file1")
    parser.add_argument("ctm_phone2frames", type=str,
                       help="Input file2")
    parser.add_argument("ctm_phone2frames_mark", type=str,
                       help="Output file")
    args = parser.parse_args()
    return args

"""
it requires two input files. respectively.
postion_dependent_phone_mark.txt is as follows:
<postion_dependent_phone> <position_independent_phone-id>
SIL 0
SIL_B 0
SIL_E 0
SIL_I 0
SIL_S 0
<sss> 1
<sss>_B 1
<sss>_E 1
<sss>_I 1
<sss>_S 1


ctm_phone2frames format is as follows:
<recode-id> <channel> <start_frame> <number of frame> <position_phone>
p226_001 1 0 53 SIL
p226_001 1 53 15 P_eng_B
p226_001 1 68 6 L_eng_I
p226_001 1 74 13 IY_eng_I
p226_001 1 87 11 Z_eng_E
p226_001 1 98 10 K_eng_B
p226_001 1 108 11 AO_eng_I
p226_001 1 119 5 L_eng_E
p226_001 1 124 12 S_eng_B
p226_001 1 136 9 T_eng_I
p226_001 1 145 6 EH_eng_I
p226_001 1 151 7 L_eng_I
p226_001 1 158 10 AH_eng_E
p226_001 1 168 58 SIL
p226_002 1 0 58 SIL
p226_002 1 58 15 AA_eng_B

ctm_phone2frames_mark format is as follows:
<recode-id> <channel> <start_frame> <number of frame> <position_independent_phone_id>
p226_001 1 0 53 0
p226_001 1 53 15 33
p226_001 1 68 6 27
p226_001 1 74 13 24
p226_001 1 87 11 45
p226_001 1 98 10 26
p226_001 1 108 11 7
p226_001 1 119 5 27
p226_001 1 124 12 36
p226_001 1 136 9 38
p226_001 1 145 6 15
p226_001 1 151 7 27
p226_001 1 158 10 6
p226_001 1 168 58 0
p226_002 1 0 58 0
p226_002 1 58 15 4


"""
# make a diction for position_dependent_phone_mark file
def get_position_dependent_phone_mark(position_dependent_phone_mark_filename):
    position_dependent_phone2mark = {}
    with open(position_dependent_phone_mark_filename, 'r') as fh:
      content = fh.readlines()
    for line in content:
      line = line.strip('\n')
      line_split = line.split()
      position_dependent_phone2mark[line_split[0]] = int(line_split[1])
    return position_dependent_phone2mark

def get_ctm_phone2frames(ctm_phone2frames_filename):
    ctm_phone2frames = {}
    with open(ctm_phone2frames_filename, 'r') as fh:
      content = fh.readlines()
    for line in content:
      line = line.strip('\n') 
      line_split = line.split()
      k = tuple([line_split[0],line_split[1],line_split[2],line_split[3]])
      v = line_split[4]
      ctm_phone2frames[k] = v

    return ctm_phone2frames

def main():
    args = get_args()
    position_dependent_phone2mark = get_position_dependent_phone_mark(args.position_dependent_phone_mark)
    ctm_phone2frames = get_ctm_phone2frames(args.ctm_phone2frames)
   
    with common_lib.smart_open(args.ctm_phone2frames_mark, 'w') as f_writer:
      for k, v in ctm_phone2frames.items():
        if v in position_dependent_phone2mark.keys():
          ctm_phone2frames[k] = position_dependent_phone2mark[v]
          # tupe to string : using join()
          print("{0} {1}".format(" ".join(k),ctm_phone2frames[k]), file=f_writer)
    
if __name__ == "__main__":
  main()


# test 
# current folder path:/home4/md510/w2019a/kaldi-recipe/nana_research
#  source-md/egs/nana_research/ctm_position_phone_mark.py exp/tri4a_phone_sub/postion_dependent_phone_mark.txt exp/tri4a_phone_sub/ctm_phone2frames exp/tri4a_phone_sub/ctm_phone2frames_mark
