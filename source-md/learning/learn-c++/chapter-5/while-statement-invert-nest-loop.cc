#include <iostream>

int main(){
    //loop between 1 and 5;
    int outercount = 5;
    while (outercount>=1){
        int innercount=outercount;
        //loop between innercount and 1;
        while(innercount>=1){
            std::cout<<innercount--<<" ";
        }
        //print newline at end of the innercount;
        std::cout<<std::endl;
        --outercount;
        
    }
    return 0;


}

//result:
//5 4 3 2 1 
//4 3 2 1 
//3 2 1 
//2 1 
//1 

