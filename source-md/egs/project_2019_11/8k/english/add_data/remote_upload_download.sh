#!/bin/bash

. path.sh
. cmd.sh
cmd="slurm.pl  --quiet --exclude=node06,node07"
steps=
. utils/parse_options.sh || exit 1


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


# Syntax of rsync command:
# Local Sync: # rsync {options} {Source} {Destination}
# Remote Sync pull: # rsync {options}  <User_Name>@<Remote-Host>:<Source-File-Dir>  <Destination>
# Remote Sync Push: # rsync  <Options>  <Source-Files-Dir>   <User_Name>@<Remote-Host>:<Destination>
# Some of the commonly used options in rsync command are listed below:

# -v, –verbose                             Verbose output
# -q, –quiet                                  suppress message output
# -a, –archive                              archive files and directory while synchronizing ( -a equal to following options -rlptgoD)
# -r, –recursive                           sync files and directories recursively
# -b, –backup                              take the backup during synchronization
# -u, –update                              don’t copy the files from source to destination if destination files are newer
# -l, –links                                   copy symlinks as symlinks during the sync
# -n, –dry-run                             perform a trial run without synchronization
# -e, –rsh=COMMAND            mention the remote shell to use in rsync
# -z, –compress                          compress file data during the transfer
# -h, –human-readable            display the output numbers in a human-readable format
# –progress                                 show the sync progress during transfer


# Remote Sync Push: # rsync  <Options>  <Source-Files-Dir>   <User_Name>@<Remote-Host>:<Destination>
# support  breakpoint resume
# rsync -rP --rsh=ssh <Source-Files-Dir>   <User_Name>@<Remote-Host>:<Destination>
if [ ! -z $step01 ];then
   # new class to hangzhou service
   rsync -rP --rsh=ssh data/train_imda_part3_raw_data.tar.gz  

   
fi
