#include <string>
#include <iostream>

//g++ -std=c++11  -fdiagnostics-color=auto -W  ~/w2019a/learn-c++/chapter-4s/string.cc -o ~/w2019a/learn-c++/chapter-4s/string.o
int main(){

    std::cout<<"You enter your name: ";
    std::string name;
    std::getline(std::cin, name);
    
    std::cout<< "You enter your age: ";
    std::string age;
    std::getline(std::cin, age);
    

    std::cout<<"Your name is "<<name<<"and age is "<<age<<std::endl;

    return 0;

    



}
