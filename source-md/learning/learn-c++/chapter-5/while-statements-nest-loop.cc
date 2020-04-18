#include <iostream>

int main(){

    int outercount = 1;
    while (outercount<=5){
    
        int innercount = 1;
        while(innercount<=outercount){
            std::cout<<innercount++<<" ";
        }
        std::cout<<std::endl;
        ++outercount;
    }

    return 0; 

}

//result:
//1 
//1 2 
//1 2 3 
//1 2 3 4 
//1 2 3 4 5 

