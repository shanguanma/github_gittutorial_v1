#!/usr/bin/env python3

# copyright 2019 Ma Duo
# Apache 2.0

import argparse
import sys
sys.path.append('steps/libs')
import common as common_lib
def get_args():
  parser=argparse.ArgumentParser(
   description="""Function of this script is using position_independent phone id 
    to mark position_dependent phone.""" )
  parser.add_argument("position_independent_phone_set", type=str,
                      help="Input file1")
  parser.add_argument("phone_map",type=str,
                      help="Input file2")
  parser.add_argument("postion_depedent_phone_mark",type=str,
                      help="Output file")
  args = parser.parse_args()
  return args

"""
    it requires two input files. respectively 
    position_independent_phone_set.txt and english_phone_map.txt.
    position_independent_phone_set.txt format is as follows:
    <position_independent_phone> <position_independent_phone-id>
    for example:
    SIL 0
    <sss> 1
    english_phone_map.txt format is as follows:
    <position_dependent_phone> <position_independent_phone>
    SIL SIL
    SIL_B SIL
    SIL_E SIL
    SIL_I SIL
    SIL_S SIL
    <sss> <sss>
    <sss>_B <sss>
    <sss>_E <sss>
    <sss>_I <sss>
    <sss>_S <sss>
    output file format is as follows:
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
"""
def get_position_independent_phone(position_independent_phone_filename):
  position_independent_phone2id = {}
  with open(position_independent_phone_filename, 'r') as fh:
    content = fh.readlines()
  for line in content:
    line = line.strip('\n')
    line_split = line.split()
    position_independent_phone2id[line_split[0]] = int(line_split[1])
  return position_independent_phone2id

def get_phone_map(phone_map_filename):
  phone_map = {}
  with open(phone_map_filename, 'r') as fh:
    content = fh.readlines()
  for line in content:
    line = line.strip('\n')
    line_split = line.split()
    phone_map[line_split[0]] = line_split[1]
  return phone_map


def main():

  args = get_args()
 
  position_independent_phone2id = get_position_independent_phone(args.position_independent_phone_set)
  phone_map = get_phone_map(args.phone_map)
  with common_lib.smart_open(args.postion_depedent_phone_mark, 'w') as f_writer:
    for phone_dependent in phone_map.keys():
      if phone_map[phone_dependent] in position_independent_phone2id.keys():
        phone_map[phone_dependent] =position_independent_phone2id[phone_map[phone_dependent]]        
        print("{0} {1}".format(phone_dependent, phone_map[phone_dependent]),file=f_writer)

if __name__ == "__main__":
  main()
 

# test
# current folder path:/home4/md510/w2019a/kaldi-recipe/nana_research
# source-md/egs/nana_research/make_id_for_position_phone.py exp/tri4a_phone_sub/position_independent_phone_set.txt exp/tri4a_phone_sub/english_phone_map.txt exp/tri4a_phone_sub/postion_dependent_phone_mark.txt
















