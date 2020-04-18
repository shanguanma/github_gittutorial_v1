#!/usr/bin/env python3

import sys
sys.path.append('source-md/egs/python_speech_features')
from python_speech_features import signal_process 
import numpy
from scipy.fftpack import dct
# fbank feature,
def fbank(signal,samplerate=16000,winlen=0.025,winstep=0.01,
          mel_filters=26,NFFT=512，lowfreq=0,highfreq=None,
          preemphasis=0.97,winfunc=lambda x: numpy.hamming((x,))):
    """
    计算fbank 特征
    :param signal:语音信号，是一个一维数组，
    :param samplerate:语音信号的采样率
    :param winlen:一帧的长度，单位是秒，默认是0.025s(25ms)
    :param winstep:帧移，单位是秒，默认是0.01s(10ms)
    :param mel_filters:梅尔滤波器组的个数，默认是26
    :param NFFT: FFT的个数，默认是512，
    :param lowfreq:梅尔滤波器最低频率，在频率空间上,默认是0,在梅尔空间上，对应的也是零
    :param highfreq:梅尔滤波器最高频率，是频率空间上最高频率/2 对应的mel频率。
    :param preemphasis:预加重系数，0表示没有预加重，默认是0.97
    :param winfunc:分析窗口，为了便于做fft变换，默认是numpy.hamming
    :return:返回两个值，第一个是numpy array ,它的尺寸是numframes * mel_filters
                        每一行包含一个特征向量，
                        第二个是每一帧的能量。
    """
    highfreq = highfreq or samplerate/2
    # 1. 对语音信号进行预加重
    # 因为高频信号的能量通常较低，因此需要增加高频部分的能量。
    # 具体来讲预加重有三个好处：增加高频部分的能量使得能量分布更加均衡；
    # 防止傅里叶变换的数值计算不稳定问题；
    # 有可能增加信噪比(Signal-to-Noise Ratio/SNR)
    signal = signal_process.preemphasis(signal,preemphasis)
    # 2. 对语音信号进行分帧和加窗处理
    frames = signal_process.frames_signal(signal, winlen*samplerate, winstep * samplerate, winfunc) 
    # 3. 对每一帧进行功率谱的计算
    power_spectrum = signal_process.power_spectrum(frames,NFFT)
    # 4. 得到每一帧能量
    energy = numpy.sum(power_spectrum,1)
    # 为类避免能量为零，对能量取对数时发生错误，
    # 所以在能量为零时，我们用一个很小的值去代替零
    energy = numpy.where(energy=0,numpy.finfo(float).eps, energy)
    # 5.得到梅尔波
    mel_bank = signal_process.get_mel_filterbank(mel_filters,NFFT,samplerate,
                                                lowfreq, highfreq)
    # 6.功率谱通过梅尔滤波器组得到fbank特征
    feat = numpy.dot(power_spectrum, mel_bank.T)
    # 和能量类似，也需要处理全是零的情况，
    feat = numpy.where(feat=0, numpy.finfo(float).eps, feat)
    return feat energy
def lifter(ceptral,L=22):
    """
    对倒谱矩阵应用一个抖动滤波器，增加dct高频系数的幅度
    :param ceptral: 倒谱矩阵，尺寸为numframes * numcep , numframes 为帧数，numcep为我们要的倒谱系数的个数，通常是13
    :param L:the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
    :return :还是一个倒谱矩阵，尺寸还是numframes * numcep , numframes 为帧数，numcep为我们要的倒谱系数的个数，通常是13
    """
    if L > 0:
        numframes, numcep = ceptral.shape()
        n = numpy.range(numcep)
        lift = 1 + (L/2.)*numpy.sin(numpy.pi*n/L)
        return lift*cepstral
    else:
        # values of L <= 0, do nothing
        return cepstral

def mfcc(signal, samplerate=16000,winlen=0.025,winstep=0.01,
         mel_filters=26,NFFT=512，lowfreq=0,highfreq=None,
         preemphasis=0.97,winfunc=lambda x: numpy.hamming((x,))
         numcep=13,cepfilter=22,appendEnergy=True):
    """ 
    计算mfcc 特征
    :param signal:语音信号，是一个一维数组，
    :param samplerate:语音信号的采样率
    :param winlen:一帧的长度，单位是秒，默认是0.025s(25ms)
    :param winstep:帧移，单位是秒，默认是0.01s(10ms)
    :param mel_filters:梅尔滤波器组的个数，默认是26
    :param NFFT: FFT的个数，默认是512，
    :param lowfreq:梅尔滤波器最低频率，在频率空间上,默认是0,在梅尔空间上，对应的也是零
    :param highfreq:梅尔滤波器最高频率，是频率空间上最高频率/2 对应的mel频率。
    :param preemphasis:预加重系数，0表示没有预加重，默认是0.97
    :param winfunc:分析窗口，为了便于做fft变换，默认是numpy.hamming
    :param numcep:the number of ceptrum  to return, default is 13
    :param cepfilter:apply a lifter to final cepstral coefficients. 0 is no lifter. Default is 22.
    :param appendEnergy: 如果是真的，我们就用这一帧的能量去取代倒谱系数的第一个系数，然后和第2到13的倒谱系数共同构成13维的mfcc特征
    :return: 是一个二维numpy.array，尺寸为:numframes * numcep, numframes 是这一个语音信号的帧数。
    """ 
    # 1. 得到fbank 特征
    feat,energy = fbank(signal, samplerate,winlen,winstep,
                        mel_filters,NFFT,lowfreq,highfreq,
                        preemphasis,winfunc)
    # 2.对fbank 特征取对数，
    feat = numpy.log(feat)
    # 3. 对fbank特征进行离散余弦变换
    feat = dct(feat, type=2,axis=1, norm='ortho')[:,:numcep]
    # 4.对倒谱矩阵进行抖动
    feat = lifter(feat,cepfilter)
    # 5.向特征只能够添加能量
    if appendEnergy:
        feat[:,0] = numpy.log(energy)
    return feat

 
# 进行一阶和二阶差分，得到39维的mfcc 特征
# 为什么要进行一阶差分和二阶差分呢？
# Deltas和Delta-Deltas通常也叫(一阶)差分系数和二阶差分(加速度)系数。
# MFCC特征向量描述了一帧语音信号的功率谱的包络信息，
# 但是语音识别也需要帧之间的动态变化信息，
# 比如MFCC随时间的轨迹，实际证明把MFCC的轨迹变化加入后会提高识别的效果。
# 因此我们可以用当前帧前后几帧的信息来计算Delta和Delta-Delta
# 计算第t帧的Delta需要t-N到t+N的系数，N通常是2。如果对Delta系数𝑑𝑡
# 再使用公式就可以得到Delta-Delta系数，这样我们就可以得到3*12=36维的特征。
# 上面也提到过，我们通常把能量也加到12维的特征里，
# 对能量也可以计算一阶和二阶差分，这样最终可以得到39维的MFCC特征向量。
def delta(feat,N):
    """
    对13维的mfcc 进行一阶差分
    :param feat:是一个二维数组，尺寸为（numframes * numcep）,numframes 是这一个语音信号的帧数。numcep倒谱个数，通常是13
    :param N:对于每一帧，计算当前帧的delta 时，需要依赖前面几帧和后面几帧，如果N=2,就表明，依赖前面两帧，和后面两帧。
    :return:一个二维数组，尺寸大小为（numframes * numcep）numframes 是这一个语音信号的帧数。numcep倒谱个数，通常是13
    """   
    if N < 1:
        raise ValueError('N must be an integer >=1')
    numframes = len(feat)
    denominator = 2 * sum([i**2 for i in range(1, N+1)])
    delta_feat = numpy.empty_like(feat)
    
