# -*- coding:utf-8 -*-

def average(a, b, c ):
    '''function to calculate the average of 3 number  '''
    return (a+b+c)/3

def calculateAverage(*args):
    """ calculates the average of n number | accepts variable length arguments"""
    argcount = len(args)
    if argcount > 0:
        sumOfNums = 0;
        for elem in args:
            sumOfNums += elem
        return sumOfNums/argcount
    else:
        return 0

def publishError(startStr, endStr, *args):
    """ publish n number of error  || accepts variable length arguments
        formal parameters """
    print(startStr)
    for elem in args:
        print("Error: ", elem)
    print(endStr)


if __name__ == "__main__":
    # calculate the average of 3 number
    avg = average(1, 2, 3)
    print("fix length argument avg :", avg)
   
    avg = calculateAverage(1,2,3,4,5,6,7,8)
    print("variable length argument avg: ", avg)


    # calculate the average of 0 number
    avg = calculateAverage()
    print("0 argument avg: ", avg)
 
    publishError("[Start]" , "[End]" , "Invalid params", "Unknown Error")
    publishError("[Start]" , "[End]" , [1, 2, 4], ("Hello", "Hi"), "Sample error")


