#!/usr/bin/env python3

# Copyright 2019  Ma Duo
# Apache 2.0

import argparse
import sys
sys.path.append('steps/libs')
import common as common_lib

def get_args():
  parser = argparse.ArgumentParser(
   description=""" This script coverts a text file(e.g: its format 
   is as follows: <segment-id> <specify speaker name>, you can also 
   view /home3/md510/w2019a/kaldi-recipe/speaker_diarization/data/test-data-2019080-kaldi/text ) 
   to a labels file(e.g: its format is as follows:<segment-id> <speaker-id>, you can also view
   data/test-data-2019080-kaldi/text),
   the speaker-id is started from one .
   e.g: source-md/egs/speaker_diarization/make_labels.py <input-file> <output-file>
   e.g. source-md/egs/speaker_diarization/make_labels.py data/test-data-2019080-kaldi/text data/test-data-2019080-kaldi/labels""")
  parser.add_argument("pre_labels",type=str,
                      help="Input file")
  parser.add_argument("labels_file",type=str,
                     help="""Output labels file, 
                     you need specify name of file,e.g:
                     $output_path/labels.""")
  args = parser.parse_args()
  return args

def main():
  args = get_args()
  # make dict for pre_labels,
  # its key is segment-id , corresponding value is speaker_name  
  seg2spk_name = {}
  speaker_name_list = [] 
  spk2id={}
  with common_lib.smart_open(args.pre_labels) as pre_labels_file:
    for line in pre_labels_file:
      segment_id, speaker_name = line.strip().split()
      seg2spk_name[segment_id] = speaker_name
      if speaker_name not in speaker_name_list:
        speaker_name_list.append(speaker_name)
    
    for i in range(len(speaker_name_list)):
      spk2id[speaker_name_list[i]] = i+1

  with common_lib.smart_open(args.labels_file,'w') as labels_file:
    for segment_id  in seg2spk_name.keys():
      speaker_name = seg2spk_name[segment_id]
      if speaker_name in spk2id.keys():
        seg2spk_name[segment_id] = spk2id[speaker_name]
      print("{0} {1}".format(segment_id,seg2spk_name[segment_id]), file=labels_file)


if __name__ == "__main__":
  main()






 
