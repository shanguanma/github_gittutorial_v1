#!/usr/bin/env python3

# Copyright 2019  Ma Duo 
#           2019  Li Wen jie
# Apache 2.0

import argparse
import sys
sys.path.append('steps/libs') 
import common as common_lib

def get_args():
  parser = argparse.ArgumentParser(
   description=""" This script converts a transcript file which is from
   textGrid to a reco2num_spk file.transcript file format is as follows:
   <recod-id> <specify speaker name>. recod-id is from wav.scp first colum.
   reco2num_spk format is as follows:
   <recod-id> <number of speakers>.
   e.g:source-md/egs/speaker_diarization/make_reco2num_spk.py <input-file>  <output-file>
   e.g: source-md/egs/speaker_diarization/make_reco2num_spk.py \
          data/test-data-2019080-tmp/transcript_pre_reco2num_spk.txt \
          data/test-data-2019080-kaldi/reco2num_spk
   for example:called the script, you can see /home3/md510/w2019a/kaldi-recipe/speaker_diarization/run_v2.sh""")	
	
  parser.add_argument("transcript",type=str,
                      help="Input transcript file")
  parser.add_argument("recod2spk_file",type=str,
                      help="Output recod2spk file")
  args = parser.parse_args()
  return args

def main():
  args = get_args()
  recod_id2speaker_name = {}
  with common_lib.smart_open(args.transcript) as transcript_file:
    for line in transcript_file:
      recod_id, speaker_name = line.strip().split()
      if recod_id not in recod_id2speaker_name.keys():
        recod_id2speaker_name[recod_id]=[speaker_name]
        
      else:
        speaker_name_list=recod_id2speaker_name[recod_id]
        if speaker_name not in speaker_name_list:
          speaker_name_list.append(speaker_name)
          recod_id2speaker_name[recod_id]=speaker_name_list
        else:
          pass 
  with common_lib.smart_open(args.recod2spk_file, 'w') as recod2spk_writer:
    for recod_id in recod_id2speaker_name.keys():
      recod_value=recod_id2speaker_name[recod_id]
      print("{0} {1}".format(recod_id,len(recod_value)), file=recod2spk_writer) 

if __name__ == '__main__':
  min()	
