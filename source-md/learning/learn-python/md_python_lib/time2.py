#!/usr/bin/env python3
# 2020 Ma Duo TL@NTU

import time


def int_to_time(second):
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    time = Time(hour, minute, second)
    return time

class Time:
    # 这里的self就是指这个类Time的实例对象，记住，就是代表本身这个类

    # __init__ is 一个特殊方法，它是对象初始时被调用。
    # 它这个方法的参数通常和这个类本身的属性使用相同的名字
    # 比如：这个方法的参数，比如是hour, 而对应这个类本身的属性self.hour 都使用相同的名字hour.

    def __init__(self, hour=0, minute=0, second=0):
        self.hour = hour
        self.minute = minute
        self.second = second
    
    # __str__ 是也是一个特殊方法，返回类对象的一个字符串形式.
    # when you print a object, Python call this method(e.g:__str__)
    def __str__(self):
        return "%.2d:%.2d:%.2d" %(self.hour,self.minute, self.second)
    
    # __add__ can do add operate on object
    #def __add__(self, other):
    #    second = self.time_to_int() + other.time_to_int()
    #    return int_to_time(second)
    # I use type-based dispatch
    def __add__(self, other):
        if isinstance(other, Time):
            return self.add_time(other)
        else:
            return self.increment(other)

    def add_time(self, other):
        seconds = self.time_to_int() + other.time_to_int()
        return int_to_time(seconds)

    def increment(self, second):
        seconds = self.time_to_int() + second
        return int_to_time(seconds)
    # 这个方法可以实现加法交换，比如，
    def __radd__(self, other):
        return self.__add__(other)

    def print_time(self):
        print("%.2d:%.2d:%.2d" %(self.hour, self.minute, self.second))
    
    def time_to_int(self):
        minute = self.hour * 60 + self.minute
        second = minute * 60 + self.second
        return second
    
    #def increment(self, second):
        # output is time
    #    second += self.time_to_int()
    #    return int_to_time(second)

    def print_attributes(self):
        for attr in vars(self):
            print(attr, getattr(self, attr))


class Point:

    def __init__(self, x=0, y=0):
        self.x = x 
        self.y = y
    def __str__(self):
        return "(%g, %g)" %(self.x, self.y)

    def __add__(self, other):
        sum_x = self.x + other.x
        sum_y = self.y + other.y
        return "(%g, %g)" %(sum_x, sum_y)

    

if __name__ == "__main__":
    start = Time()
    start.hour = 9
    start.minute = 45
    start.second = 0.0
    start.print_time()
    end = start.increment(1750)
    print(end)
    #end.print_time()

    point = Point()
    start1 = Time(9,45)
    duration = Time(1,35)
    print("two time are added, then result is :",start1 + duration)
    print("two time are added, then result is :",start1 + 1337)
    print("call __radd__, two time are added, then result is :",1337 + start1)
    #print(start1 + duration)

    t1 = Time(7, 43)
    t2 = Time(7, 41)
    t3 = Time(7, 37)
    sum = sum([t1, t2, t3])
    print(sum)

    print(t1.print_attributes())



    