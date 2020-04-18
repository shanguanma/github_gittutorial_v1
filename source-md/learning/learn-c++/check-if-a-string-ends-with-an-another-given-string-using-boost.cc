//std::string endsWith() using Boost Library
//Boost's algorthm library provides implementation of both case sensitive and insensitive  
//implementation of endsWith() function for string

//g++ check-if-a-string-ends-with-an-another-given-string-using-boost.cc -o check-if-a-string-ends-with-an-another-given-string-using-boost.o
#include <iostream>  //std::cout
#include <string>
#include <boost/algorithm/string.hpp> 

int main()
{
    std::string mainStr = "This is a sample String";
    std::string toMatch = "String";
    
    // test case sensitive implementation of endsWith() function
    bool result = boost::algorithm::ends_with(mainStr, toMatch);
    std::cout<<result<<std::endl;
    
    //test case insensitive implementation of endsWith() function
    result = boost::algorithm::iends_with(mainStr, "striNG");
    std::cout<<result<<std::endl;

    return 0;

}
