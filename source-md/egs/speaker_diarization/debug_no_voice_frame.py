#!/usr/bin/env python3


import kaldi_io
import argparse
import numpy as np
import sys
sys.path.append('diarization')
import VB_resegmentation
# Load the MFCC features

def get_args():
  parser = argparse.ArgumentParser(
   description=""" this script is used to debug no vioce frame in test data.""")
  parser.add_argument("data_dir",type=str,
                       help="the path of features of data")
  parser.add_argument('init_rttm_filename', type=str,
                       help='The rttm file to initialize the VB system, usually the AHC cluster result')
  args = parser.parse_args()
  return args

def get_utt_list(utt2spk_filename):
  with open(utt2spk_filename, 'r') as fh:
    content = fh.readlines()
  utt_list = [line.split()[0] for line in content]
  print("{} utterances in total".format(len(utt_list)))
  return utt_list

# prepare utt2num_frames dictionary
def get_utt2num_frames(utt2num_frames_filename):
  utt2num_frames = {}
  with open(utt2num_frames_filename, 'r') as fh:
    content = fh.readlines()
  for line in content:
    line = line.strip('\n')
    line_split = line.split()
    utt2num_frames[line_split[0]] = int(line_split[1])
    return utt2num_frames

def create_ref(uttname, utt2num_frames, full_rttm_filename):
  num_frames = utt2num_frames[uttname]

  # We use 0 to denote silence frames and 1 to denote overlapping frames.
  ref = np.zeros(num_frames)
  speaker_dict = {}
  num_spk = 0

  with open(full_rttm_filename, 'r') as fh:
    content = fh.readlines()
  for line in content:
    line = line.strip('\n')
    line_split = line.split()
    uttname_line = line_split[1]
    if uttname != uttname_line:
      continue
    start_time, duration = int(float(line_split[3]) * 100), int(float(line_split[4]) * 100)
    end_time = start_time + duration
    spkname = line_split[7]
    if spkname not in speaker_dict.keys():
      spk_idx = num_spk + 2
      speaker_dict[spkname] = spk_idx
      num_spk += 1
      for i in range(start_time, end_time):
        if i < 0:
          raise ValueError("Time index less than 0")
        elif i >= num_frames:
          print("Time index exceeds number of frames")
          break
        else:
          if ref[i] == 0:
            ref[i] = speaker_dict[spkname]
          else:
            ref[i] = 1 # The overlapping speech is marked as 1.
  return ref.astype(int)


def main():
  args = get_args()
  feats_dict = {}
  for key,mat in kaldi_io.read_mat_scp("{}/feats.scp".format(args.data_dir)):
    feats_dict[key] = mat
  utt_list = get_utt_list("{}/utt2spk".format(args.data_dir))
  utt2num_frames = get_utt2num_frames("{}/utt2num_frames".format(args.data_dir))
  
  for utt in utt_list:
    # Get the alignments from the clustering result.
    # In init_ref, 0 denotes the silence silence frames
    # 1 denotes the overlapping speech frames, the speaker
    # label starts from 2.
    init_ref = create_ref(utt, utt2num_frames, args.init_rttm_filename)

    # load MFCC features
    X = (feats_dict[utt]).astype(np.float64)
    #assert len(init_ref) == len(X)
    print("init_ref length:",len(init_ref))
    print("test data length:",len(X))



if __name__ == "__main__":
  main()

