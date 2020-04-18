//How to remove Substrings from a String in C++
//std::string class provides a member function string::erase() 
//to remove some characters from a given position i.e.
//npos 就是相对pos的偏移量，也就是substring的长度
//string& erase (size_t pos = 0, size_t len = npos);

//It accepts a position and length of characters to be deleted from that position. 
//It removes those characters from string object and also returns the updated string.

#include <iostream>  //std::cout
#include <string>    //std::string::npos, std::basic_string::erase, std::basic_string::find
#include <algorithm> //std::for_each
#include <functional> // std::bind,std::ref
#include <vector>
/*
 * erase first occurrence of given substring from main string
 */

using namespace std;

void eraseSubstr(std::string & mainStr, const std::string & toErase)
{
    //search for the substring in string
    size_t pos = mainStr.find(toErase);
    if (pos != std::string::npos)
    {
        //if found then erase it from main string
        mainStr.erase(pos, toErase.length());

    }

}

/*
 *erase all occurrence of given substring from main string
 */
void eraseAllSubstr(std::string & mainStr, const std::string & toErase)
{
    size_t pos = std::string::npos;
    //search for the substring in main string in a loop utill nothing is found
    while((pos = mainStr.find(toErase)) != std::string::npos)
    {
        //if found then erase it from main string
        mainStr.erase(pos, toErase.length());
    }

}

/*
 * erase all occurrences of all given substrings from main string using c++11 stuff
 */
void eraseAllSubstrhighway(std::string &mainStr, const std::vector<std::string> & strList )
{
    //Iterator over the given list of substings, FOr each substring call eraseAllSubstr() to 
    //remove its all occurrences from main string.
    //template <class InputIterator, class Function>
    //   Function for_each (InputIterator first, InputIterator last, Function fn);
    //std::bind()
    //std::ref()
    std::for_each(strList.begin(), strList.end(), std::bind(eraseAllSubstr, std::ref(mainStr), std::placeholders::_1));
}
/*
 *earse all ocuurrences of all given substring from main string using pre c++ stuff
 */

void eraseSubStrPre(std::string & mainStr, const std::vector<std::string> & strList)
{
    //Iterate over the given list of substring. for each substring call eraseAllSubstr()
    //to remove its all occurrences from main string.
    for (std::vector<std::string>::const_iterator it = strList.begin(); it != strList.end(); it++)
    {
        eraseAllSubstr(mainStr, *it);

    }

}


int main()
{
    std::string str = "Hi this is a sample string this is for is testing is.";
    eraseSubstr(str, "this");
    std::cout<<"正在测试函数eraseAllstr::"<< str<< std::endl;
    eraseSubStrPre(str, {"for", "is", "test"});
    std::cout<<"正在测试函数eraseSubStrPre::"<<str<<std::endl;
    eraseAllSubstrhighway(str, {"for", "is", "testing"});
    std::cout<<"正在测试函数 eraseSubStrhighway::"<<str<<std::endl;
    
    return 0;

}












