#include <iostream>

int main(){
    int outer{ 1 };
    while (outer<=5)
    {
        int inner{ outer -5 };
        while (inner <= 0 )
        {
            std::cout<<" ";
            ++inner;
        }
        while(inner <= outer )
        {
            std::cout<< inner++ << " ";

        }
        std::cout<< std::endl;
        ++outer;
    }
    return 0;
}


//result:
//      1
//     1 2
//    1 2 3
//   1 2 3 4
//  1 2 3 4 5
