//in this article we will discuss both case sensitive ans insensitive implementations to 
//check if a string ends with an another given string,

//in c++, std::string class does not provides any ensWith() function to check if a string ends with 
//an another given string. let's see how to do that using std::string::compare()
//and Boost Library.
//
// g++ --std=c++11 check-if-a-string-ends-with-an-another-given-string.cc -o check-if-a-string-ends-with-an-another-given-string.o
#include <string>    //std::compare
#include <algorithm> //std::all_of
#include <iostream>  //std::cout
#include <iterator>  //std::next


//idea:
//std::string class provides a member function compare() with different overloaded versions. 
//We will use one of its overloaded version i.e.
//
//int compare (size_t pos, size_t len, const string& str) const;
//
//It accepts a string as argument to match, starting position for match and number 
//of characters to match. If string matches then it returns 0 else returns > 0 or < 0 based 
//on difference in ascii values of first unmatched character.
//
//To check if a main string ends with a given string, we should look into last n characters 
//only for main string, where n is the size of given string. Let’s use std:string::compare() to 
//find the last occurrence of given string from position (Size of Main string – size of given string).



/*
 * case sensitive implementation of endsWith() using std::compare from <string> 
 * It check if the string 'mainStr' ends with given string 'toMatch'
 */
bool endsWith(const std::string &mainStr, const std::string &toMatch)
{
    if(mainStr.size() > toMatch.size() &&
         mainStr.compare(mainStr.size() - toMatch.size(), toMatch.size(), toMatch) == 0)
        return true;
    else
        return false;
}

/*
 * case sensitive implementation of endsWith() using std::all_of() from <algorithm>
 * It checks if the string "mainStr" ends with given string "toMatch"
 *
 * std::next(position, number of element position offset(1 by default))
 * std::all_of(first positon , final posotion, pred)
 * pred:一元函数，接受范围内的元素作为参数，并返回可转换为bool的值。
 */

bool endsWith_secApproach(const std::string &mainStr, const std::string &toMatch)
{
    auto it = toMatch.begin();
    return mainStr.size()>=toMatch.size() &&
           std::all_of(std::next(mainStr.begin(), mainStr.size() - toMatch.size()), mainStr.end(), [&it](const char &c){return c == *(it++);
        });
}


/*
 * case insensitive implementation of endWith() using std::all_of() from <algorithm>
 * It check if the string "mainStr" ends with given string "toMatch"
 */
bool endsWithCaseInsensitive(std::string mainStr, std::string toMatch)
{
    auto it = toMatch.begin();
    return mainStr.size() >=toMatch.size() &&
        std::all_of(std::next(mainStr.begin(), mainStr.size() - toMatch.size()), mainStr.end(), [&it](const char &c){ return ::tolower(c) == ::tolower(*(it++));

    });

}

int main()
{
    std::string mainStr = "This is a sample String";
    std::string toMatch = "String";
    
    //test case sensitive implementation of endsWIth() function
    bool result = endsWith(mainStr, toMatch);
    std::cout<<result<<std::endl;
    //test casr insensitive implementation of endsWith() function
    result = endsWith_secApproach(mainStr, toMatch);
    std::cout<<result<<std::endl;
    //test case insensitive implementation of endsWith() function
    result = endsWithCaseInsensitive(mainStr, toMatch);
    std::cout<<result<<std::endl;
    return 0;


}











