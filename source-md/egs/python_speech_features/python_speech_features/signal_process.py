#!/usr/bin/env python3
# code refrence:https://github.com/jameslyons/python_speech_features/blob/master/python_speech_features/sigproc.py
# documnet refrence : http://fancyerii.github.io/books/mfcc/
# Ma Duo 2019

"""
note: 提取mfcc 特征时，对语音信号处理可以分为如下几步：
1. 预加重
2. 分帧后加窗操作
3. 对每一帧信号进行离散傅里叶变换（DFT），得到周期图的功率图估计。
4. 这时对每一帧的功率谱进行求和得到这一帧的能量，为将来向12维的mfcc在第一维添加这一帧的能量做准备。
   如果NFFT是512的话，那么每一帧的功率谱为257个点构成，
   如果这257个点都是零的话，他们相加还是零，这时候能量为0,对能量取log 就会出问题，
   所以用一个很小的数字来代替零，这是在code 的层面的考虑的，具体可以看这个函数的操作fbank()
5. 对周期图功率谱进行梅尔滤波。（这是一组大约20-40(通常26)个三角滤波器组，
   它会对上一步得到的周期图的功率谱估计进行滤波。我们的滤波器组由26个(滤波器)长度为257的向量组成，
   每个滤波器的257个值中大部分都是0，只有对于需要采集的频率范围才是非零。
   输入的257点的信号会通过26个滤波器，我们会计算通过每个滤波器的信号的能量。）,这时就得到了fbank 特征，它是一个二维数组，
   每一行是一个向量，长度就等于梅尔滤波的个数，如果梅尔滤波为26，那么数组的列数为26，数组的行数就是帧数，（这个音频划分了多少帧，就是那个帧数）

5. 对上面的26点个信号能量取log
6. 对上面取过log 的26 点的进行离散余弦变换（DCT）,得到26个倒谱系数(Cepstral Coefficients)，
   最后我们保留2-13这12个数字，这12个数字就叫MFCC特征。可以看出从step3开始我们的操作都是对每一帧信号进行
   的，因此如果mfcc 的是按13维计算的话，那么第一维应该是能量，而这个能量是信号的每一帧的能量，并且这个能量也是
   取对数的，
7. 对上面的13维mfcc 取一阶差分和二阶差分，得到39维的mfcc特征,
   (Deltas和Delta-Deltas通常也叫(一阶)差分系数和二阶差分(加速度)系数。
   MFCC特征向量描述了一帧语音信号的功率谱的包络信息，
   但是语音识别也需要帧之间的动态变化信息，比如MFCC随时间的轨迹，
   实际证明把MFCC的轨迹变化加入后会提高识别的效果。
   因此我们可以用当前帧前后几帧的信息来计算Delta和Delta-Delta,计算第t帧的Delta需要t-N到t+N的系数，N通常是2。也就是
   对于当前帧来说，考虑前两帧和后两帧。
"""

import decimal  # 十进制的算法模块

import numpy
import math
import logging

# 1. 预加重
def preemphasis(signal, coeff=0.97):
    """
    这个函数对信号进行预加重(Pre-Emphasis)。
    因为高频信号的能量通常较低，因此需要增加高频部分的能量。
    具体来讲预加重有三个好处：增加高频部分的能量使得能量分布更加均衡；
    防止傅里叶变换的数值计算不稳定问题；
    有可能增加信噪比(Signal-to-Noise Ratio/SNR)。它的计算公式为：
    𝑦𝑡=𝑥𝑡−𝛼𝑥𝑡−1
    
    :param signal : input signal of preemphasis fiter.
    :param coeff: The preemphasis coefficient. 0 is no filter, default is 0.95.
    :return :the filtered signal. 
    """
    # # 第一个时刻不需要处理；后面的是当前信号减去coeff乘以上一个时刻的信号
    return numpy.append(signal[0],signal[1:] - coeff * signal[:-1])


# 2. 分帧后加窗
# 首先确保每一帧的采样数数一个十进制的整数
def round_half_up(number):
    """
    确保每一帧的采样数数一个十进制的整数
    >>> decimal.Decimal('1.3145').quantize(decimal.Decimal('1.000'))
    Decimal('1.314')
    >>> decimal.Decimal('1.3145').quantize(decimal.Decimal('1.00'))
    Decimal('1.31')
    >>> decimal.Decimal('1.3145').quantize(decimal.Decimal('1.0'))
    Decimal('1.3')
    >>> decimal.Decimal('1.3145').quantize(decimal.Decimal('1'))
    Decimal('1') 
    
    >>> decimal.Decimal('1.6145').quantize(decimal.Decimal('1.000'),rounding=decimal.ROUND_HALF_UP)
    Decimal('1.615')
    >>> decimal.Decimal('1.6145').quantize(decimal.Decimal('1.00'),rounding=decimal.ROUND_HALF_UP)
    Decimal('1.61')
    >>> decimal.Decimal('1.6145').quantize(decimal.Decimal('1.0'),rounding=decimal.ROUND_HALF_UP)
    Decimal('1.6')
    >>> decimal.Decimal('1.6145').quantize(decimal.Decimal('1'),rounding=decimal.ROUND_HALF_UP)
    Decimal('2')

    :param number : type str,
    :return : an integer 
    """
    return int(decimal.Decimal(number).quantize(decimal.Decimal('1'), rounding=decimal.ROUND_HALF_UP))

# 其次要制作一个滑动窗口，这样分帧的效率快一点，分帧时不用这个函数也是可以,
# 但是我还没有完全弄懂，所以宁愿先不用。
#def rolling_window(a, window,step=1):
    # reference: https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html
    #            https://jessicastringham.net/2017/12/31/stride-tricks/
    
#    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
#    strides = a.strides + (a.strides[-1],)
#    return numpy.lib.stride_tricks.as_strided(a, shape=shape,strides=strides)[::step]

           
# 进行分帧后加窗操作
def frames_signal(signal,frame_len, frame_step,winfunc=lambda x:numpy.one((x,))):
    """
    把信号变成有重叠的帧
    
    :param signal: 语音信号
    :param frame_len: 一帧中有多少个采样点
    :param winlen: 一帧的长度，单位秒，默认0.025s (25毫秒)
    :param winstep: 帧移，单位秒，默认0.01s (10毫秒)
    :param winfunc: 窗函数，比如使用汉明窗：winfunc=numpy.hamming, 
                    矩形窗：winfunc=numpy.one 也就是值全为1，
                    但是它的缺点是从1突然变成0会造成尖峰。比较常见的是汉明窗 
    :returns: 一个数组，每一个元素是一帧的数据，大小是(NUMFRAMES, frame_len)
    
    """
    # 本例子的这个音频为采样频率为8000，音频的长度为2479616，也就是说这个音频有这么多个整数点构成，
    # 信号长度 = 2479616
    signal_len = len(signal)
    # 帧长 =0.025 * 8000=200
    frame_len = int(round_half_up(frame_len))
    # 帧移 = 0.01 * 8000=80
    frame_step = int(round_half_up(frame_step))
    if signal_len <= frame_len:
        numframes = 1
    else:
        # 帧数为（30994）
        numframes = 1 + math.ceil((1.0 * signal_len - frame_len) / frame_step)) 
    # 不足一帧按一帧计算，所以要对音频最后进行补零操作
    # padding 后的总的点数 为（2479640）
    padlen = int((numframes - 1) * frame_step + frame_len)
    # padding的点
    zeros = numpy.zeors((padlen - signal_len,))
    # 原始信号signal_len 加上padding 的点
    padsignal = numpy.concatenate((signal_len, zeros))
    
    # 进入真正的分帧操作，
    # 我要使用numpy.tile(),下面是它的用法
    """
    >>> c = numpy.array([1,2,3,4])
    # (4,)的向量
    >>> numpy.tile(c,(4,1))
    # 结果是二维的，先broadcasting成(1, 4)，
    # 然后复制得到(4, 4)
    array([[1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4],
           [1, 2, 3, 4]])
    """
    # numpy.arange(0,frame_len) = [0,1,...,199] ,是一个（200,）数组，通过tile变成(30994,200)一个二维数组，
    # 每一行都是(0,1,...,199),总共30994行，
    # (0, numframes*frame_step, frame_step) 是一个(30994,)的数组，数组的内容是(0,80,160,...,30994x80),
    # 经过tile之后变(200,30994),经过转置，得到(30994,200),那么每一列都是(0,80,160,...,30994x80),总共200列
    # 那么他们两个相加尺寸还是(30994,200)
    # 你会发现，它的第一行就是第一帧信号，第二行就是第二帧，...
    indices =  numpy.tile(numpy.arange(0,frame_len),(numframes,1)) +
               numpy.tile((0, numframes*frame_step, frame_step),(frame_len,1)).T
    #print(indices.dtype) # int64
    #indices = numpy.array(indices, dtype=numpy.int32)
    frames = padsignal[indices]
    # 然后用winfunc(frame_len)得到200个点的窗函数，类似的用tile变成(30994,200)的win，
    # 最后把frames * win得到加窗后的分帧信号，其shape还是(30994, 200)。
    # 默认的窗函数是方窗，也就是值全为1，但是它的缺点是从1突然变成0会造成尖峰。比较常见的是汉明窗
    win = numpy.tile(winfunc(frame_len),(numframes,1)) 
    frames = frames * win 
    return frames 

# 3.计算功率谱
# 首先计算幅度谱
def magnitude_spectrum(frames, NFFT):
    """
    计算每一帧的幅度谱，这里的frames 就是frames_signal() 函数的输出，如果frames 是一个N X D 矩阵，
    那么这个函数的输出尺寸应该是N x (NFFT/2 + 1)
    N 是帧数，D 就是每一帧的帧长，如果帧长为0.025s,信号抽样频率为8000，那么D就是0.025*8000=200.
    幅度谱计算公式：|fft(xi)|
    :param frames: it is 2d array, per row is one frame data.
    :param NFFT: it is fft length. if NFFT>frame_len: the frames are zero-padding.
    :return :If frames is an NxD matrix, output will be Nx(NFFT/2+1). 
             Each row will be the magnitude spectrum of the corresponding frame.
    """
    if numpy.shape(frames)[1] > NFFT:
        logging.warn(
          " frame lenght (%d) is greater than fft size (%d) ,frame will be truncated. Increase NFFT to avaid.",
           numpy.shape(frames)[1],NFFT)
    # 复数值
    complex_specturm = numpy.fft.rfft(frames,NFFT)
    return numpy.abs(complex_spectrum)

# 开始计算功率谱
def power_spectrum(frames, NFFT):
    """
    计算每一帧的功率谱，这里的frames 就是frames_signal() 函数的输出，如果frames 是一个N X D 矩阵，
    那么这个函数的输出尺寸应该是N x (NFFT/2 + 1)
    N 是帧数，D 就是每一帧的帧长，如果帧长为0.025s,信号抽样频率为8000，那么D就是0.025*8000=200.
    功率谱计算公式：p = |fft(xi)|^2 / NFFT
    :param frames: it is 2d array, per row is one frame data.
    :param NFFT: it is fft length. if NFFT>frame_len: the frames are zero-padding.
    :return :If frames is an NxD matrix, output will be Nx(NFFT/2+1). 
             Each row will be the power spectrum of the corresponding frame.
    
    """

    return numpy.square(magnitude_spectrum(frames,NFFT)) / NFFT

# 5. 构造梅尔滤波，得到梅尔谱

# 首先进行梅尔频率和频率进行转换

def hz2mel(hz):
    """
    Convert a value in Hertz to Mels.

    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * numpy.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

# 开始构造梅尔滤波器，得到梅尔滤波器组
def get_mel_filterbank(mel_filters=26,NFFT=512,samplerate=16000,lowfreq=0,highfreq=None):
    """
    计算一个梅尔谱，mel filters are stored in the rows, the columns corresponding to fft bins ,
    the filters are returned as an array size mel_filters * (NFFT//2 + 1).

    :param mel_filters: the number of the mel_filters in the filtbank,
    :param NFFT: the size of fft, default is 512,
    :param samplerate:the sample rate of the signal in hz case, affect mel space,
    :param lowfreq:lowest band edge of mel filters,default is 0 hz,
    :param highfreq:hightest band edge of mel filters,default is samplerate/2
    :return :a numpy array ,its size:mel_filters * (NFFT//2 + 1) 
    """ 
    highfreq = highfreq or samplerate/2
    assert highfreq <= samplerate/2, 'highfreq is greater than samplerate/2'

    # 1. covert hz to mel space
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    # 2. 在mel 空间上平均分配他们，
    melpoints = numpy.linspace(lowmel,highme, mel_filters + 2)
    # 3. 把这些mel 空间上的点转换到频率。
    hzpoints = mel2hz(melpoints)
    # 4. 把这些频率对应最近接近的的FFT的bin里
    # 因为FFT的频率没办法精确的与上面的频率对应，因此我们把它们对应到最接近的bin里
    bin=numpy.floor(NFFT+1) * hzpoints/samplerate
    mel_bank = numpy.zeros([mel_filters, NFFT//2 +1])
    for j in range(0,mel_filters):
        for i in range(int(bin[j]),int(bin[j+1])):
            mel_bank[j,i] = (i -bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            mel_bank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return mel_bank


