#include <iostream>

//int main(){
    // outercount is row
//    int outercount = 1;
//    while (outercount<=5){
        //innercount is colunm
//        int innercount=5;
//        while (innercount>=1){
        
//            if (innercount<=outercount)
//                std::cout<<innercount<<" "; //" " is one space
//            else
//                std::cout<<" "; //"  "is one space.
//            --innercount;
//        }
//        std::cout<<std::endl;
//        ++outercount;
//    }
//    return 0;
//}


//result:
//        1
//       2 1
//      3 2 1
//     4 3 2 1
//    5 4 3 2 1


int main(){

    int outercount = 1;
    while (outercount<=5){

        int innercount = 5;
        while(innercount>=1){

            if (innercount<=outercount)
                std::cout<<innercount<<" "; //" " is one space
            else
                std::cout<<"  "; //"  "are two spaces.
            --innercount; 

        }
        std::cout<<std::endl;
        ++outercount;
    }
    return 0;
}


//result:
//          1
//        2 1
//      3 2 1
//    4 3 2 1
//  5 4 3 2 1
//
