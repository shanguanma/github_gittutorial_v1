import kaldiio
d = kaldiio.load_scp('wav_10.scp')
for key in d:
    rate, numpy_array = d[key]
    print(key,rate,len(numpy_array))



import numpy
from kaldiio import WriteHelper
with WriteHelper('ark,t:file.ark') as writer:
    for i in range(10):
        print(type(numpy.random.randn(5, 10)))
        writer(str(i), numpy.random.randn(5, 10))


#d = kaldiio.load_ark('file_2.ark')  # d is a generator object
#for key, numpy_array in d:
#    print("first: "+key, numpy_array[0])


# === load_ark can accepts file descriptor, too
#with open('file.ark') as fd:
#    for key, numpy_array in kaldiio.load_ark(fd):
#        print("second: " + key, numpy_array)
print(numpy.load('exp/tri4a_phone_sub_npy/p226_001:27-feats.npy')[0])
