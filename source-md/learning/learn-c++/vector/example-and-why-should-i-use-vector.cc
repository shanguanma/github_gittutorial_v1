#include <iostream>
#include <vector>
using namespace std;
//Important Points about std::vector :
//1.) Ordered Collection:
//In std::vector all elements will remain in same order in which they are inserted.
//
//2.) Provides random access:
//Indexing is very fast in std::vector using opeartor [], just like arrays.
int main(){
    //Creating a vector object without any initialization 
    //will create an empty vector with no elements i.e.
    //This is a vector of int
    std::vector<int> vecOfInts;
    
    // While adding it automatically adjust it't size
    for (int i = 0; i < 10; i++)
        vecOfInts.push_back(i);
    std::vector<int>::iterator it = vecOfInts.begin();
    while (it != vecOfInts.end())
    {
        std::cout<<*it<<" , ";
        it++;

    }
    std::cout<<std::endl;
    for (int i = 0; i < vecOfInts.size(); i++)

        std::cout<<vecOfInts[i]<<" , ";
    std::cout<<std::endl;
    return 0;
}
