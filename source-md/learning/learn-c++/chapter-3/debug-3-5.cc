#include <iostream>
#include "plog/Log.h" // plog is open source library
//how to run the script
//g++ -std=c++11 -I plog/include chapter-3/debug-3-5.cc -o chapter-3/debug-3-5.o
//chapter-3/debug-3-5.o
int getUserInput(){

    //std::cerr <<"getUserInput() is called \n";
    LOGD <<"getUserInput() is called \n";
    std::cout << "Enter a number: ";
    int x;
    std::cin >> x;
    return x;
}

int main(){
    plog::init(plog::debug, "Logfile.txt"); //this log file is generated at current work contents
    //plog::init(plog::debug, "chaper-3/debug_logfile,txt"); this is not work.
    LOGD << "main() is called \n";
    //std::cerr << "main() is called \n";

    int x = getUserInput();
    std::cout << "You entered: " << x <<std::endl;
    return 0;

}
