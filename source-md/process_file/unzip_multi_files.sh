#!/bin/bash
#查询压缩包出来，然后解压
find /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/test -name  *.tar.gz > test/find.log
find /home4/md510/w2019a/kaldi-recipe/project_2019_11/8k/english/add_data/test -name  *.tgz >> test/find.log
#data=$(cat ls.log)
#for i in $data;do
for i in `cat test/find.log`;do
     tar -zxf $i >> /dev/null 
done
