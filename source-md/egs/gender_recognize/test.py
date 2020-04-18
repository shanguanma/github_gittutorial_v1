#!/usr/bin/env  python3
import json

personDict ={
 'bill':'tech',
 'federer':'tennis',
 'woods':'golf',
 'ali':'boxing'
}

with open('person.json','w') as json_file:
    json_file.write(json.dumps(personDict, indent=4, ensure_ascii=False, sort_keys=True))


d = {} # 一个普通的字典
d.setdefault('a', []).append(1)
d.setdefault('a', []).append(2)
d.setdefault('b', []).append(4)
print (d)
