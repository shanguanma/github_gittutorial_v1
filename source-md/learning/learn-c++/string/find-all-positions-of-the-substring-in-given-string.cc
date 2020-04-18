#include <iostream>
#include <algorithm>
#include <vector>

/*
 * find all positions of the a substring in given string
 */
using namespace std;
void findAllOccurrences(std::vector<size_t> & vec, std::string data, std::string toSearch){    
     //get the first occurence 
     size_t pos = data.find(toSearch);
     
     //repeat till end is reached.
     while(pos != std::string::npos)
     {   
         //add positon to the vector
         vec.push_back(pos);
         //get the next occurrence from the current position
         pos = data.find(toSearch, pos + toSearch.size());


     }

}

int main(){
    std::string data = "Hi this is a Sample string, 'IS' is here 3 times";
    std::vector<size_t> vec;
    // get all occurrences of the 'is' in the vector 'vec'
    findAllOccurrences(vec, data, std::string("is"));
    std::cout<<"All index position of 'is' in given string are,"<<std::endl;

    for(size_t pos : vec)
        std::cout<<pos<<std::endl;
    return 0;


}


