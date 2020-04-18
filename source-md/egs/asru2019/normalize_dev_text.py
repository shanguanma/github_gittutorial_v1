#!/usr/bin/env python3

import argparse
import re
def get_args():
    parser=argparse.ArgumentParser(
     description= """ remove [S],[T],[P],[N] and so on in text.""")
    parser.add_argument('text',type=str,help="input text file")
    args = parser.parse_args()
    return args



def main():
    args=get_args()
    with open (args.text,'r') as f:
       for line in f:
           line = line.strip('\n')
           line_list = line.split()
           #print(line_list[0],' '.join(line_list[1:]))
           str_= ' '.join(line_list[1:])
           # remove [S],[T],[P],[N]
           l_ = re.sub(u'\\[.*?]', '',str_)
           # remove ?,.，。？《》
           l_ = re.sub(u'\?|\,|\.|\，|\。|\？|\《|\》', '', l_)
           # covert upper to lower
           l_ =l_.lower()
           # covert '-' to ' '
           l_ = re.sub(u'\-',' ', l_)
           # remove ~,@
           l_ = re.sub(u'\~|\@','', l_)
           #l_ = re.sub(u'\^' '', '',l_)
           print('{0} {1}'.format(str(line_list[0]),l_))
           #print('{0:<15} {1:^10}'.format(str(line_list[0]),l_))

if __name__ == "__main__":
    main()
    #s="我是一个人(中国人)aaa[真的]bbbb{确定}"
    #a = re.sub(u"\\(.*?\\)|\\{.*?}|\\[.*?]", "", s)
    #print (a )   
    # 我是一个人aaabbbb 

