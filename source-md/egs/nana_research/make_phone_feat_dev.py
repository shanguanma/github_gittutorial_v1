#!/usr/bin/env python3

# copyright 2019 Ma Duo
# Apache 2.0
import argparse
import sys
sys.path.append('steps/libs')
import common as commom_lib
import numpy as np
#from kaldiio import WriterHelper
from kaldiio import WriteHelper
def get_args():
    parser=argparse.ArgumentParser(
     description="""Function of this script that I used ctm_phone_frames_mark to 
      get phone matrix, the row of the phone matrix is how many frames the phones lasts
      the column of the phone matrix is one-hot vector of the phone.
      one-hot vector lenght is postion_indepent_phone set,for example: 
      exp/tri4a_phone_sub/position_independent_phone_set.txt is postion_indepent_phone set.
      exp/tri4a_phone_sub/ctm_phone2frames_mark is ctm_phone_frames_mark. its format detail 
      you can see in the script(e.g:source-md/egs/nana_research/ctm_position_phone_mark.py)
      """)

    parser.add_argument("ctm_phone_frames_mark", type=str,
                        help="Input file")
    #parser.add_argument("phone_matrix", type=str,
    #                    help="Output file")

    args = parser.parse_args()
    return args

"""
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
# get a dict, its format is {'p226_001': [(53, 0), (15, 33), (6, 27), (13, 24), (11, 45), (10, 26), (11, 7), (5, 27), (12, 36), (9, 38), (6, 15), (7, 27), (10, 6), (58, 0)], 'p226_002': [(58, 0), (15, 4)]} 
def get_ctm_phone_frames_mark(ctm_phone_frames_mark_filename):
    ctm_phone_2frames_mark = {}
    #value_list = []
    with open(ctm_phone_frames_mark_filename,'r') as fh:
        content = fh.readlines()
    for line in content:
        line = line.strip('\n') 
        recode_id, channel, start_frame, frames, phone_id = line.split()
        #value_list = [tuple([line_split[3],line_split[4]])]
        if recode_id in ctm_phone_2frames_mark:
            ctm_phone_2frames_mark[recode_id].append(tuple([int(frames), int(phone_id)]))
        else:
            ctm_phone_2frames_mark[recode_id]=[tuple([int(frames), int(phone_id)])]
    return ctm_phone_2frames_mark


#  
def make_specified_dim_one_hot(position, dim=46):
    # one-hot 
    array= np.zeros(dim)
    array[position] = 1
    # disturbed one-hot
    #array = np.random.rand(dim)*0.000000001
    #array[position] = 1
    return array    
    
# >>> a = np.array([0,1,0,0,0])
# >>> np.tile(a,(3,1))
# array([[0, 1, 0, 0, 0],
#        [0, 1, 0, 0, 0],
#        [0, 1, 0, 0, 0]])
# >>> np.tile(a,(3,2))
# array([[0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0],
#        [0, 1, 0, 0, 0, 0, 1, 0, 0, 0]])
def repeat_one_hot(array,num_frames):
    return np.tile(array,(num_frames,1))

         
def main():
    args = get_args()
    #ctm_phone_2frames_mark = get_ctm_phone_frames_mark(args.ctm_phone_frames_mark)
    #print(ctm_phone_2frames_mark)
    #with open(args.ctm_phone_frames_mark,'r')as f, WriteHelper('ark,t:file_2.ark') as writer:
    with open(args.ctm_phone_frames_mark,'r')as f, WriteHelper('ark,t,scp:file_2.ark,file_2.scp') as writer:
        for line in f:
            line = line.strip('\n')
            recode_id, channel, start_frame, frames, phone_id = line.split()
            writer(str(recode_id + ':' + phone_id),repeat_one_hot(make_specified_dim_one_hot(int(phone_id),dim=46),int(frames)))
        



if __name__ == '__main__':
   main()
