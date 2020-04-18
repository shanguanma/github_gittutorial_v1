#include <iostream>
//g++ -std=c++11 -fdiagnostics-color=auto -W chapter-4/print-chars-as-integers-via-type-casting.cc -o chapter-4/print-chars-as-integers-via-type-casting.o
int main(){
    char ch{'b'};
    //method 1
    //int i{ch};
    //std::cout<<ch<<std::endl;
    //std::cout<<i<<std::endl;
    //method 2
    std::cout<<ch<<std::endl;
    std::cout<<static_cast<int>(ch)<<std::endl;
    
    return 0;
}
