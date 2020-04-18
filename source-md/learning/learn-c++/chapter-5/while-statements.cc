#include <iostream>

//question: we want to do something every n iterations,
// such as print a newline.

int main(){
    
    int count = 1;
    
    while (count<=50){
        // print the number (pad number under 10 with a leading 0 for formattingpurposes)
        if (count<10){
         
            std::cout<<"0"<<count<<" ";
        }
        else
            std::cout<<count<<" ";
        // if the loop vaiable is divisible by 10, print a newline
        if (count % 10 == 0){

            std::cout<<std::endl;
        }
        ++count;
        }
        return 0;
}
    
//result:
//01 02 03 04 05 06 07 08 09 10 
//11 12 13 14 15 16 17 18 19 20 
//21 22 23 24 25 26 27 28 29 30 
//31 32 33 34 35 36 37 38 39 40 
//41 42 43 44 45 46 47 48 49 50 

