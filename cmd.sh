# you can change cmd.sh depending on what type of queue you are using.
# If you have no queueing system and want to run on a local machine, you
# can change all instances 'queue.pl' to run.pl (but be careful and run
# commands one by one: most recipes will exhaust the memory on your
# machine).  queue.pl works with GridEngine (qsub).  slurm.pl works
# with slurm.  Different queues are configured differently, with different
# queue names and different ways of specifying things like memory;
# to account for these differences you can create and edit the file
# conf/queue.conf to match your queue's configuration.  Search for
# conf/queue.conf in http://kaldi-asr.org/doc/queue.html for more information,
# or search for the string 'default_config' in utils/queue.pl or utils/slurm.pl.

#export train_cmd="slurm.pl --quiet --nodelist=node06"
export train_cmd="slurm.pl --quiet --exclude=node01,node02,node06,node08"
#export decode_cmd="slurm.pl --quiet --nodelist=node05"
#export decode_cmd="slurm.pl --quiet --nodelist=node06"
#export decode_cmd="slurm.pl --quiet --nodelist=node07"
#export decode_cmd="slurm.pl --quiet --exclude=node01,node02,node04,node05,node07"
export decode_cmd="slurm.pl --quiet --exclude=node07,node08"

# the use of cuda_cmd is deprecated, used only in 'nnet1',
#export cuda_cmd="slurm.pl --quiet --gres=gpu:1"
#export gpu_train_cmd="slurm.pl --quiet --gres=gpu:1"
#export cuda_cmd="slurm.pl --quiet --nodelist=node05 --gres=gpu:1"
export cuda_cmd="slurm.pl --quiet --exclude=node01,node02,node03"
#export cuda_cmd="slurm.pl --quiet --nodelist=node07"
export gpu_train_cmd="slurm.pl --quiet"



# JHU setup
#export train_cmd="queue.pl --mem 2G"
#export cuda_cmd="queue.pl --mem 2G --gpu 1 --config conf/gpu.conf"
#export decode_cmd="queue.pl --mem 4G"
