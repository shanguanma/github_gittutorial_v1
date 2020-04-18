#include <iostream>
#include <string>
#include <boost/algorithm/string.hpp>

using namespace std;
int main()
{
    std::string mainStr = "This is a sample string";
    std::string toMatch = "This";
    
    //test case insensitve implementation of startsWith function
    bool result = boost::algorithm::starts_with(mainStr, toMatch);
    std::cout<<result<<std::endl;
    // test case insensitive implementation of startsWith function
    result = boost::algorithm::istarts_with(mainStr, toMatch);
    std::cout<<result<<std::endl;
    return 0;
    

}
