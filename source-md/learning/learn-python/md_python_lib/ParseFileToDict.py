#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import unicode_literals
# This function parses a file and pack the data into a dictionary
# It is useful for parsing file like wav.scp, utt2spk, text...etc
# learn from steps/data/reverberate_data_dir.py
import argparse, sys,  os
import subprocess
# below two line command are deprecated 
import imp
data_lib = imp.load_source('dml', "steps/data/data_dir_manipulation_lib.py")

#import importlib
#data_lib = importlib.util.find_spec('dml', "steps/data/data_dir_manipulation_lib.py")
# we add required arguments as names arguments for readability
def CheckArgs(args):
    if not os.path.exists(args.input_dir):
        raise Exception("must have a input file")
    return args
def GetArgs():
    parser = argparse.ArgumentParser(description='this function parses a file'
                                                 'and pack the data into a' 
                                                 'dictionary. It is useful for'
                                                 'parsing file like wav.scp,utt2spk'
                                                 'text...etc.')
                                     #formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('input_dir',help='Input data directory')
    # echo command line to stderr for logging.
    #print(subprocess.list2cmdline(sys.argv), file=sys.stderr)
    #  print(' '.join(sys.argv)) is same function as print(subprocess.list2cmdline(sys.argv), file=sys.stderr)
    print(' '.join(sys.argv))  # print command information
    print(subprocess.list2cmdline(sys.argv), file=sys.stderr) # print command information 
    args = parser.parse_args()
    args = CheckArgs(args)
    return args


def ParseFileToDict(file, assert2fields = False, value_processor = None):
    if value_processor is None:
        value_processor = lambda x : x[0]
    dict = {}
    for line in  open(file, 'r', encoding='utf-8'):
        parts = line.split()
        if assert2fields:
            assert(len(parts) == 2)
        dict[parts[0]] = value_processor(parts[1:])
    return dict



def RunKaldiCommand(command, wait = True):
    """ Runs commands frequently seen in Kaldi scripts. These are usually a
        sequence of commands connected by pipes, so we use shell=True """
    p = subprocess.Popen(command, shell=True,
                         stdout = subprocess.PIPE,
                         stderr = subprocess.PIPE)
    if wait:
        [stdout, stderr] = p.communicate()
        if p.returncode is not 0:
            raise Exception("There was an error while running the command {0}\n".format(command)+'-'*10+'\n'+stderr)
        return stdout, stderr
    else:
        return p
 
# echo command line to stderr for logging.
#print(subprocess.list2cmdline(sys.argv), file=sys.stderr)
#  print(' '.join(sys.argv)) is same function as print(subprocess.list2cmdline(sys.argv), file=sys.stderr)
def main():
    args = GetArgs()
    # parse the utt2spk to get a dictionary
    utt2spk = ParseFileToDict(args.input_dir + "/utt2spk", value_processor = lambda x: " ".join(x))
    wav_scp = ParseFileToDict(args.input_dir + "/wav.scp", value_processor = lambda x: " ".join(x))
    print("utt2spk: ", utt2spk)
    print("wav_scp: ", wav_scp)
    if not os.path.isfile(args.input_dir + "/reco2dur"):
        print("Getting the duration of the recording.... ")
        data_lib.RunKaldiCommand("utils/data/get_reco2dur.sh {} ".format(args.input_dir))
        #RunKaldiCommand("utils/data/get_reco2dur.sh {} ".format(args.input_dir))

    durations = ParseFileToDict(args.input_dir + "/reco2dur", value_processor = lambda x: float(x[0]))
    print("durations:", durations)
if __name__ == "__main__":
    main()

