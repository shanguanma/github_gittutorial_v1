#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
#set -e
#set -u
set -o pipefail


# set run environment
cmd="/home4/md510/package/slurm.pl --quiet --nodelist=node04"  # you can set a node 
                                                               # for example : set two nodes node06, node07, you can set 
                                                               # cmd="/home4/md510/package/slurm.pl --quiet --nodelist=node06,node07" or 
                                                               # cmd="/home4/md510/package/slurm.pl --quiet --quiet --exclude=node01,node02,node3,node04,node05,node08" 
#. path.sh   # it is some python packages you need for python to run. 
. source-md/path_example.sh #      it is some python packages you need for python to run.
                            #
steps=1
. ./utils/parse_options.sh || exit 1;  # it is used to parse parameter. this file is from kaldi/egs/wsj/s5/utils/parse_options.sh
                                       # for example : steps is a parameter. 

steps=$(echo $steps | perl -e '$steps=<STDIN>;  $has_format = 0;
  if($steps =~ m:(\d+)\-$:g){$start = $1; $end = $start + 10; $has_format ++;}
        elsif($steps =~ m:(\d+)\-(\d+):g) { $start = $1; $end = $2; if($start == $end){}elsif($start < $end){ $end = $2 +1;}else{die;} $has_format ++; }
      if($has_format > 0){$steps=$start;  for($i=$start+1; $i < $end; $i++){$steps .=":$i"; }} print $steps;' 2>/dev/null)  || exit 1

if [ ! -z "$steps" ]; then
  for x in $(echo $steps|sed 's/[,:]/ /g'); do
    index=$(printf "%02d" $x);
    declare step$index=1
  done
fi

# run script
if [ ! -z $step01 ];then
   echo "run start"
   # log/train.log is a log file, the log file for your python script(e.g: sourc-md/example_python.py)
   # CUDA_VISIBLE_DEVICES=0 is used to set GPU ID in your specify node.
   $cmd log/train.log \
   CUDA_VISIBLE_DEVICES=0 \
   python3   source-md/example_python.py   # you can set 0,1,2,3 ,other number isn't be set.
   echo "run finish"
fi
# how to run this script ?
# -o is this shell script run log file .
# sbatch   -o example.log  source-md/example_python.sh  
