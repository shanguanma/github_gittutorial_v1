#!/usr/bin/env python3

import scipy.io.wavfile as wav
(rate,sig) = wav.read('source-md/egs/python_speech_features/taot.wav')
# rate is sample of the wave file.
# sig is  2479616 lenght of 1d numpy arrary, represented speech signal.
print((rate,sig))
print((rate,len(sig)))

import librosa
signal, sample_rate = librosa.core.load("source-md/egs/python_speech_features/taot.wav")

print(signal,sample_rate)
print((len(signal),sample_rate))
print("signal size is " + str(signal.shape[0]) )
