#include <iostream>

int main(){

    char mychar = 'a';
    while (mychar<='z'){

        std::cout<<"mychar "<<static_cast<int>(mychar)<<" ";
        if (static_cast<int>(mychar) % 10 == 0){
            std::cout<<std::endl; 
        }
        ++mychar;
   }
   return 0;
}

//result:
//mychar 97 mychar 98 mychar 99 mychar 100 
//mychar 101 mychar 102 mychar 103 mychar 104 mychar 105 mychar 106 mychar 107 mychar 108 mychar 109 mychar 110 
//mychar 111 mychar 112 mychar 113 mychar 114 mychar 115 mychar 116 mychar 117 mychar 118 mychar 119 mychar 120 
//mychar 121 mychar 122
