
def time_to_int(t1):
    assert valid_time(t1)
    minute = t1.hour * 60 + t1.minute
    second = minute * 60 + t1.second
    return second

def int_to_time(second):
    # divmod 是用第一个参数除以第二个参数并以 元组的形式返回商和余数
    minute, second = divmod(second, 60)
    hour, minute = divmod(minute, 60)
    return hour, minute, second

assert x == time_to_int(int_to_time(x))

def add_time(t1, t2):
    assert valid_time(t1) and valid_time(t2)
    sum = time_to_int(t1) + time_to_int(t2)
    return(int_to_time(sum))

def valid_time(t):
    # this assume hour can more than 24
    if t.hour < 0 or t.minute < 0 or t.second < 0:
        return False
    if t.minute > 60 or t.second > 60:
        return False
    return True


def increment(t1, second):
    assert  valid_time(t1)
    sum = time_to_int(t1) + second
    return(int_to_time(sum))

def mul_time(time, factor):
    assert valid_time(time)
    mul = time_to_int(time) * factor
    return int_to_time(mul)



