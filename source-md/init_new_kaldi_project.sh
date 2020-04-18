#!/bin/bash

source_project_dir=/home4/md510/w2020/kaldi-recipe/egs/malay_cts
source_dir=/home4/md510/package/source-md
new_project=$1

echo $new_project
if [ -z $new_project ]; then
  new_project=new_project_`date "+%Y%m%d%H%M"`
fi

pk_list="cmd.sh conf path.sh  steps utils"
mk_list="kaldi_data log test"
lk_list="source-md"

[ -d $new_project ] || mkdir -p $new_project
echo "Make a new project : $new_project"

# copy tools folder from source dir
for pk in $pk_list; do
  cp $source_project_dir/$pk  $new_project -r
  echo "Copy $pk from $source_project_dir done."
done

# mkdir new folder
for mk in $mk_list; do
  [ -d $new_project/$mk ] || mkdir -p $new_project/$mk
  echo "Make new folder $new_project/$mk done."
done

# make soft link
for lk in $lk_list; do
  [ -d $new_project/$lk ] || ln -s $source_dir $new_project/$lk
  echo "Link $lk done."
done

echo -e "##Initial a new projects done. @ `date`\n"


