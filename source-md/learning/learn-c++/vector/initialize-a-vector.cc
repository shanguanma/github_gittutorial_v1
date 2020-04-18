#include <iostream>
#include <vector>
#include <list>
#include <iterator>
#include <string>

using namespace std;
// g++ --std=c++11 initialize-a-vector.cc -o initialize-a-vector.o
void example_1(){
    //creating a vector object without any initialization will 
    //creat an empty vector with no elements i.e.
    //std::vector<int> vecOfInts;
    //Initialize vector with 5 integers
    //default value of all 5 ints will be 0
    std::vector<int> vecOfInts(5);
    for (int x : vecOfInts)
        std::cout<<x<<std::endl;

}


void example_2(){

    //initialize a vector by filling similar copy of an element
    // many times we want to initialize a vector with an element 
    // of particular value instead of default value.
    // for that vector provides an overloaded constructor i.e.
    // vector(size_type n, const value_type & val, 
    //        const allocator_type & alloc = allocator_type());
    // it accepts the size of vector and an element as an argument.
    // Then it initialize the vector with n elements of value val.
    // initialize vector to 5 string objects with value "Hi"
    std::vector<std::string> vecOfStr(5, "Hi");
    for (std::string str : vecOfStr)
        std::cout<<str<<std::endl;
}


void example_3(){

     //initialize a vector with a array
     //what if we want to initialize a vecor with an array of elements
     //for the vector provides an overloaded constructor i.e.
     //vector (InoutIterator first, InputIterator last, 
     //        const allocator_type & alloc = allocator_type());
     // it accepts a range as an argument i.e two iterators and initializes 
     // the vector with elements in range(first, last] i.e. from first till
     // last -1.
     // we will using the same overloaded constructor to initialize a 
     // vector of string from an array of string i.e
     // create an array of string object
     std::string arr[] = {"first","second","third","forth"};
     //initialize vector with a string array
     std::vector<std::string> vecOfStr(arr, 
                                       arr + sizeof(arr)/ sizeof(std::string));
     for (std::string str : vecOfStr)
         std::cout<<str<<std::endl;

}

void example_4(){

    //initialize a vector with std::list
    // we will use the same overloaded constructor of std::vector to
    // initialize a vector with range i.e.
    // vector(InputIterator first,
    //     InputIterator last, const allocator_type & alloc = allocator_type());
    // This time range will be of std::list's iterator i.e.
    // creat an std::list of 5 string objects
    std::list<std::string> listOfStr;
    listOfStr.push_back("first");
    listOfStr.push_back("second");
    listOfStr.push_back("third");
    listOfStr.push_back("fourth");
    // initialize a vector with std::list
    std::vector<std::string> vecOfStr(listOfStr.begin(), listOfStr.end());
    for (std::string str: vecOfStr)
        std::cout<<str<<std::endl;
    
    //initailize a vector with other string vector object
    //std::vector<std::string> vecOfStr3(vecOfStr);
    //for (std::string str : vecOfStr3)
    //    std::cout<<str<<std::endl;

}

void example_5(){

    //initialize a vector with an other vector
    //vector provides a constructor that receives other
    //vector as an argument and initializes the current vector
    //with the copy of all elements of provided vector i.e.
    // vector(const vector & x);
    std::vector<std::string> vecOfStr;
    vecOfStr.push_back("first");
    vecOfStr.push_back("second");
    vecOfStr.push_back("third");
    //initialize a vector with other string object
    std::vector<std::string> vecOfStr3(vecOfStr);

}


int main(){
    example_1();
    example_2();
    example_3();
    example_4();
    example_5();
    return 0;

}
