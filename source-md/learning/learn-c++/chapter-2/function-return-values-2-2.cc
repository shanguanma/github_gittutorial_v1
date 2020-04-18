#include <iostream>

int getValueFromUser()
{
    std::cout<<"Enter a integer number :";
    int input{};
    std::cin>>input;
    return input;

}



int main(){

    //std::cout<<"Enter a number :";
    //int num{};
    //std::cin >>num;
     int  num{ getValueFromUser()};
    //print the value double
    std::cout << num <<"doubled is :"<< num * 2<<std::endl;
    return 0;

}



