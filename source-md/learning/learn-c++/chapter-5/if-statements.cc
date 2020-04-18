#include <iostream>

enum class ErrorCode{

    SUCCESS = 0,
    NEGATIVE_NUMBER = -1,

};

ErrorCode dosomething(int value){

    if(value<0)
        return ErrorCode::NEGATIVE_NUMBER;

    return ErrorCode::SUCCESS;
}

int main(){
 
    std::cout<<"please enter a positive number: ";
    int x;
    std::cin>> x;

    if (dosomething(x) == ErrorCode::NEGATIVE_NUMBER)
    {
        std::cout<<"you enter a negative number,you should enter a positive number"<<std::endl;
    }
    else
    {
        std::cout<<"It worked!"<<std::endl;


    }

}
