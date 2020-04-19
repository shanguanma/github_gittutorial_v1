# How to install KALDI：

` step1:`
`### download kaldi master from github`
`$ git clone https://github.com/kaldi-asr/kaldi.git`

# step2:
$ cd kaldi/tools
### check some dependent package
./extras/check_dependencies.sh

# step3:
### compile fst etc.
make -j 15

# step4:
### switch to src folder, src folder is the main place to store kaldi code.
cd ../src
### compile kaldi code, you need to specify math libriary and cuda folder path.
./configure  --mathlib=ATLAS --use-cuda --cudatk-dir=/cm/shared/apps/cuda10.0/toolkit/10.0.130 

# step5:
### clean some dependent.
make clean -j 15
### Build code dependency
make depend -j 15
### compile kaldi code
make -j 15

# step6:
### Check if kaldi's cuda is compiled successfully
cd cudamatrix
make test -j 10




# My script is stored and updated.

# A asr example:
$ source-md/w2020/kaldi-recipe/egs/malay_cts/kaldi_asr_v1.sh


# how to run it ?
$ source-md/w2020/kaldi-recipe/egs/malay_cts/run_16k_1a.sh 

# how to prepared data?

You can see librispeech in kaldi or mini_librispeech in kaldi.
their data is open source. you can download free.

for example:https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/run.sh . There, stage 1 and stage2 are data download and data preparation
