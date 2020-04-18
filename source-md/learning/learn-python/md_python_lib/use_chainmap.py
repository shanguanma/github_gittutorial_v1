"""
ChainMap
ChainMap可以把一组dict串起来并组成一个逻辑上的dict。ChainMap本身也是一个dict，但是查找的时候，会按照顺序在内部的dict依次查找。

什么时候使用ChainMap最合适？举个例子：应用程序往往都需要传入参数，参数可以通过命令行传入，可以通过环境变量传入，还可以有默认参数。我们可以用ChainMap实现参数的优先级查找，即先查命令行参数，如果没有传入，再查环境变量，如果没有，就使用默认参数。

下面的代码演示了如何查找user和color这两个参数：
"""

from collections import ChainMap
import os, argparse

# 构造缺省参数
defaults = {
    'color': 'red',
    'user': 'guest'
}


# 构造命令行参数：
parser = argparse.ArgumentParser()
parser.add_argument('-u', '--user')
parser.add_argument('-c', '--color')
namespace = parser.parse_args()
command_line_args = { k: v for k, v in vars(namespace).items() if v}

# 组合成ChainMap:
combined = ChainMap(command_line_args, os.environ, defaults)

# 打印参数：
print('color=%s' % combined['color'])
print('user=%s' % combined['user'])



### 没有任何参数，打印出默认参数：
# $ python3 use_chainmap.py
# color=red
# user=guest

### 当传入命令行参数时，优先使用命令行参数：
# $ python3 use_chainmap.py -u bob
# color=red
# user=bob

# 同时传入命令行参数和环境变量时，命令行参数的优先级别较高：
# $ user=admin color=green python3 use_chainmap.py -u bob
# color=green
# user=bob 
