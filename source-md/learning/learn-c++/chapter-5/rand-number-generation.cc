#include <iostream>
#include <cstdlib> // for std::rand() and std::srand()
#include <ctime> //for std::time()

int main()
{
    // set initial seed value to system clock
    std::srand(static_cast<unsigned int>(std::time(nullptr))); 
    for (int count=1; count <100; ++count)
    {
        std::cout<<std::rand()<< "\t";
        if (count % 5 == 0)
         {

            std::cout<<std::endl;
         }

    }
    return 0;
}
