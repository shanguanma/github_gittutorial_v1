# -*- coding:utf-8 -*-


# base class
class Animal(object):
    def run(self):
        print("Animal is running....")
   

# subclass

class Dog(Animal):
    def run(self):
        print('Dog is running....')

class Cat(Animal):
    def run(self):
        print('Cat is running....')

class Tortoise(Animal):
    def run(self):
        print('Tortoise is running slowly ....')
class Timer(object):
    def run(self):
        print('Start ....')



def run_twice(Animal):
    Animal.run()
    Animal.run()


dog = Dog()
dog.run()

cat = Cat()
cat.run()

# instance class
run_twice(Dog())
run_twice(Cat())
run_twice(Tortoise())

run_twice(Timer())
