#include <iostream>
int add(int x, int y) //x and y are created and enter scope here
{
    //x and y are visible/usable within this function only
    return x + y;

}//y and x go out of scope and are destoryed here.

int main()
{
    int a{5}; //a is created, initialized, and enters scope here
    int b{6}; //b is created,initialized, and enters scope here
    //a and b are visible/ usable within this function only
    std::cout<<add(a, b)<<std::endl; //calls function add() with x = 5, y = 6
    return 0;

}//b and a go out of scope and are destoryed here.


// let’s trace through this program in a little more detail. The following happens, in order:
//
// execution starts at the top of main
// main‘s variable a is created and given value 5
// main‘s variable b is created and given value 6
// function add is called with values 5 and 6 for arguments
// add‘s variable x is created and initialized with value 5
// add‘s variable y is created and initialized with value 6
// operator+ evaluates expression x + y to produce the value 11
// add copies the value 11 back to caller main
// add‘s y and x are destroyed
// main prints 11 to the console
// main returns 0 to the operating system
// main‘s b and a are destroyed

