#!/usr/bin/env python3

# copyright 2019 Ma Duo
# Apache 2.0

import argparse
import sys
sys.path.append('steps/libs')
import common as common_lib
import os
def get_args():
  parser = argparse.ArgumentParser(
   description=""" This script makes kaldi text from kaldi text scp format
    text.scp format is as follows:
    <record-id> <transcription-path>
    note: <record-id> is first column of wav.scp, <transciption-path> is corresponding to text path.
    kaldi text format is as follow:
    <record-id> <transcription>
    note: <record-id> is first column of wav.scp, <transciption> is corresponding to text.""")

  parser.add_argument("kaldi_text_scp", type=str,
                      help="Input file")
  parser.add_argument("kaldi_text",type=str,
                     help="output file")
  args = parser.parse_args()
  return args

def main():
  args = get_args()
  kaldi_text_scp={}
  with common_lib.smart_open(args.kaldi_text_scp) as f:
    for line in f:
      record_id, transcription_path = line.strip().split()
      kaldi_text_scp[record_id] = transcription_path

  with common_lib.smart_open(args.kaldi_text, 'w') as kaldi_text_writer:
    for record_id in kaldi_text_scp.keys():
      transcription_path=kaldi_text_scp[record_id]
      with common_lib.smart_open(transcription_path) as f1:
     
        line = f1.readline().strip()
       
        # get file name with suffix , for example:p256_058.txt
        tempfilename=os.path.basename(transcription_path)
        file_name, extention=os.path.splitext(tempfilename)
        
      print("{0} {1}".format(file_name, line), file=kaldi_text_writer) 


if __name__ == "__main__":
  main()

# test
# current path:/home4/md510/w2019a/kaldi-recipe/nana_research
# source-md/egs/nana_research/make_text.py 10_text 10_text_1
