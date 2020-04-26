
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
# My script is stored and updated. :blush:
***
**Data detail is as follows:**
```
   train set : kaldi_data/train_clean_100
   test set : kaldi_data/{dev_clean  dev_other  test_clean  test_other}
   dict:     kaldi_data/dict_nosp
```


# How to install KALDI ?

<details><summary>expand</summary><div>
   
   
```shell
 *Step1:*

### download kaldi master from github


$ git clone https://github.com/kaldi-asr/kaldi.git

*step2:*

$ cd kaldi/tools

### check some dependent package

$ ./extras/check_dependencies.sh

*step3:*

### compile fst etc.

$ make -j 15

*step4:*

### switch to src folder, src folder is the main place to store kaldi code.

$ cd ../src

### compile kaldi code, you need to specify math libriary and cuda folder path.
### for example: use ATLAS math libriary and cuda vesion = 10

$ ./configure  --mathlib=ATLAS --use-cuda --cudatk-dir=/cm/shared/apps/cuda10.0/toolkit/10.0.130

*step5:*

### clean some dependent.

$ make clean -j 15

### Build code dependency

$ make depend -j 15

### compile kaldi code

$ make -j 15

*step6:*

### Check if kaldi's cuda is compiled successfully

$ cd cudamatrix

$ make test -j 10

### in order to use n-gram command to build lm, you must install srilm tools.
$ cd kaldi/tools

$ cp -r srilm.tgz ./

$ install_srilm.sh 
```
</div></details>


```
### A asr example:

$ source-md/w2020/kaldi-recipe/egs/librispeech_demo/kaldi_asr_v1.sh

***

### How to run it ?

$ source-md/w2020/kaldi-recipe/egs/librispeech_demo/run_16k_1a.sh 

### How to prepared data?

You can see librispeech in kaldi or mini_librispeech in kaldi.

their data is open source. you can download free.

for example:https://github.com/kaldi-asr/kaldi/blob/master/egs/librispeech/s5/run.sh . 

There, stage 1 and stage2 are data download and data preparation
```
