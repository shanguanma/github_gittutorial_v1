#include <iostream>
#include <string>
#include <algorithm>

int main(){
   std::string data = "hi this is a sample string.";
   
   //iterator over all the characters in a string
   //and call a lambda function on each of them
   std::for_each(data.begin(), data.end(), [](char &c){
        static int last = ' ';
        if(last == ' ' && c != ' ' && ::isalpha(c))
        c = ::toupper(c);
        last = c;
   });
   std::cout<<data<<std::endl;
}


