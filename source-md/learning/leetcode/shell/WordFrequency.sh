# Source : https://leetcode.com/problems/word-frequency/

################################################################################## 
# 
# Write a bash script to calculate the frequency of each word in a text file words.txt.
# 
# For simplicity sake, you may assume:
# 
# words.txt contains only lowercase characters and space ' ' characters.
# Each word must consist of lowercase characters only.
# Words are separated by one or more whitespace characters.
# 
# For example, assume that words.txt has the following content:
# the day is sunny the the
# the sunny is is
# 
# Your script should output the following, sorted by descending frequency:
# 
# the 4
# is 3
# sunny 2
# day 1
# 
# Note:
# Don't worry about handling ties, it is guaranteed that each word's frequency count is unique.
# 
# [show hint]
# Hint:
# Could you write it in one-line using Unix pipes?
##################################################################################

#!/bin/bash

# Read from the file words.txt and output the word frequecy list to stdout.
#  tr '[:upper:]' '[:lower:]' 把大写字母变成小写字母 
cat words.txt | tr [:space:] "\n" | sed '/^$/d' | tr '[:upper:]' '[:lower:]'|sort|uniq -c| sort -nr | awk '{ print $2,$1}'


# cat words.txt | tr [:space:] "\n" 把空格改成换行符，也就把words.txt的内容变成
# The
# day
# is
# Sunny
# the
# the
# The
# Sunny
# is
# is


# awk '{$1=""; print;}' file_3.txt |uniq -d

#呃 十 一 号 十 月 哦 我 跟 你 讲 一 个 东 西 but 不 可 以 讲 出 去
#ah final saturday

# $2 is set second column 
# the command is used to sum second column
# awk '{ sum += $2 } END { print sum }' wordfrenquency.txt  



# how to compute total duration by using segments file.

#awk '{ sum += $4-$3 } END { print sum/3600 }' test_segments


