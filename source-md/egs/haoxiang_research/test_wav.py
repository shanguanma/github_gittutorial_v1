#!/usr/bin/env python3

import librosa
signal, sample_rate = librosa.core.load("/home/haoxiang215/workspace/audio/test_src5/out/nnet_iter77_trloss10.8143_trsnr23.1938_valoss11.2258_vasnr22.9359-320013/p232_001.wav.pr.wav", sr=16000)

print(signal, sample_rate)

import soundfile as sf
sf.write('p232_001.wav.pr.wav', signal, sample_rate, subtype='PCM_16')

#import sys
#import array
#import struct
#import wave
#
#def convert(fin, fout, chunk_size = 1024 * 1024):
#    chunk_size *= 4    # convert from samples to bytes
#
#    waveout = wave.open(fout, "wb")
#    waveout.setparams((1, 2, 44100, 0, "NONE", ""))
#
#    while True:
#        raw_floats = fin.read(chunk_size)
#        if raw_floats == "":
#            return
#        floats = array.array('f', raw_floats)
#        samples = [sample * 32767
#                   for sample in floats]
#        raw_ints = struct.pack("<%dh" % len(samples), *samples)
#        waveout.writeframes(raw_ints)
#
#convert(open(sys.argv[1], "rb"), open(sys.argv[2], "wb"))
