#include <string>
#include <iostream>
#include <algorithm>

/*
 * case sensitive implementation of startwith()
 * it check if the string 'mainstr' start with given string 'toMatch'
 */

bool startsWith(std::string mainStr, std::string toMatch)
{
    //std::string::find return 0 if toMatch is found at starting
    if(mainStr.find(toMatch) == 0)
        return true;
    else
        return false;

}


/*
 * case insensitive implementation of startWith()
 * It checks if the string 'mainstr' starts with given string 'toMatch'
 */

bool startsWithCaseInsensitive(std::string mainStr, std::string toMatch)
{
    //convert mainStr to lower case
    std::transform(mainStr.begin(), mainStr.end(), mainStr.begin(), ::tolower);
    //convert toMatch to lower case
    std::transform(toMatch.begin(), toMatch.end(), toMatch.begin(), ::tolower);
    if (mainStr.find(toMatch) == 0)
        return true;
    else
        return false;



}

int main()
{
    std::string mainStr = "This is the sample string";
    std::string toMatch = "This";
    //test case sensitive implementation of startsWith function
    
    bool result = startsWith(mainStr, toMatch);
    std::cout<<result<<std::endl;
    //test case insensitive implementation of startsWith function
    result = startsWithCaseInsensitive(toMatch, "this");
    std::cout<<result<<std::endl;
    return 0;
}


