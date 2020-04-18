# -*- coding:utf-8 -*-

class Student(object):
    # __init__ is special method, in order to tie Attributes to the Student class.
    def __init__(self, name, gender):
        self.name = name
        self.__gender = gender
    def get_gender(self):
        return self.__gender
    def set_gender(self,gender):
        if gender == 'male' or 'female':
            self.__gender = gender
        
        else: 
            raise ValueError('Bad gender!')

bart = Student('Tom','male')
if bart.get_gender() != 'male':
    print('test is failure!')

else:
    bart.set_gender('female')
    if bart.get_gender() != 'female':
        print('test is failture!')
    else:
        print('test is successful!')
