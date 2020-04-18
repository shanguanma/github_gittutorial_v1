#include <iostream>
// g++ -std=c++11 -W ~/w2019a/learn-c++/chapter-4s/enumerator.cc -o ~/w2019a/learn-c++/chapter-4s/enumerator.o
enum Color{
    COLOR_BLACK, //assigned 0
    COLOR_RED, // assigned 1
    COLOR_BLUE, // assigned 2
    COLOR_GREEN, // assigned 3
    COLOR_WHITE, // assigned 4
    COLOR_CYAN, // assigned 5
    COLOR_YELLOW, // assigned 6
    COLOR_MAGENTA, // assigned 7


};
// It is possible to explicitly define the value of enumerator. 
// These integer values can be positive or negative and can share
// the same value as other enumerators. Any non-defined enumerators 
// are given a value one greater than the previous enumerator.
enum Animal{
    ANIMAL_CAT = -3,
    ANIMAL_DOG, // assigned -2
    ANIMAL_PIG, // assigned -1
    ANIMAL_HORSE = 5, 
    ANIMAL_GIRAFFE = 5, // shares same value as ANIMAL_HORSE
    ANIMAL_CHICKEN, // assigned 6



};


int main(){

    Color paint(COLOR_WHITE);
    std::cout<<paint<<std::endl;  //result is 4
    int mypet = ANIMAL_PIG;
    std::cout<<ANIMAL_HORSE<<std::endl;
    // the follow will cause compiler error
    //Color color;
    //std::cin >> color;
    //
    //One workaround is to read in an integer, 
    //and use a static_cast to force the compiler 
    //to put an integer value into an enumerated type:
    int inputColor;
    std::cin>> inputColor;
    Color color = static_cast<Color>(inputColor);
}   
