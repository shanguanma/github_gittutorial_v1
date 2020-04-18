//finding duplicates in a vector
//steps are:
//create a map of <string, int> type to store the frequency cout of each string in vector
//iterate over all the elements in vector try to insert it in map as key with value as 1,
//if string already exists in map then increment its value by 1
//
// g++ -std=c++11 find-duplicates-in-a-vector-string.cc -o find-duplicates-in-a-vector-string.o
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <algorithm>
#include <functional>

//print the contents of vector
template <typename T>
void print(T & vecOfElements, std::string delimeter = " , ")
{
    for(auto & elem : vecOfElements)
        std::cout<<elem<<delimeter;
    std::cout<<std::endl;

}

/*
 * generic function to find dumplicates elements in vector.
 * It adds the dumplicate elements and their duplication count in given map countMap
 */

template <typename T>
void findDuplicates(std::vector<T> & vecOfElements, std::map<T, int> countMap)
{
    //iterate over the vector and store the frequency of each element in map
    for(auto & elem : vecOfElements)
    {
        auto result = countMap.insert(std::pair<std::string, int>(elem, 1));
            if(result.second == false)
                result.first->second++;
    }
    // remove the element from Map which has 1 frequency count
    for(auto it = countMap.begin(); it != countMap.end();)
    {
        if(it->second == 1)
            it = countMap.erase(it);
        else
            it++;
    }

}

int main()
{
    //vector of strings
    std::vector<std::string> vecOfStrings{"at", "hello", "hi", "there", "where", "now", "is", \
                                          "that", "hi", "where", "at", "no", "yes", "at"};

    print(vecOfStrings);
    // creat a map to store the frequency of each element in vector
    std::map<std::string, int> countMap;
    // interate over the vector and store the frequency of each element in map
    for(auto & elem : vecOfStrings)
    {
        auto result = countMap.insert(std::pair<std::string, int>(elem, 1));
        if(result.second == false)
            result.first->second++;


    }
    std::cout<<"Duplicate elements and their duplication count "<<std::endl;
    // iterate over the map
    for(auto & elem : countMap)
    {
        // if frequency count is greater than 1 then its a duplicate element
        if(elem.second > 1 )
        {
            std::cout<<elem.first<<"::"<<elem.second<<std::endl;
          

        }
    }
    /*
     * finding duplicate in vector using generic function
     */

     std::map<std::string, int> duplicateElements;
     //get the duplicate elements in vector
     findDuplicates(vecOfStrings, duplicateElements);
     std::cout<<"Duplicate elements and their duplication count "<<std::endl;
     //for(auto & elem : duplicateElements)
     //{
     //    std::cout<<elem.first<<" :: "<<elem.second<<std::endl;
     //}
     return 0;
 
}
