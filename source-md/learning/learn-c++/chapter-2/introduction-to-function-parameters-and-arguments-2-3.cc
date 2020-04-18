//we learned that we could have a function return a value back to
//the function's caller.
//We used that to create a modular getValueFromUser function 
//that we used in this program:
//
//g++ --std=c++11 introduction-to-function-parameters-and-arguments-2-3.cc -o introduction-to-function-parameters-and-arguments-2-3.o
//#include <iostream>
//int getValueFromUser(){
    //std::cout<<"Enter A number: ";
    //int input{ };
    //std::cin>>input;
    //return input;


//}


//int main(){

    //int num = { getValueFromUser() };
    //std::cout<<num<<"double is : "<< num * 2 <<std::endl;
    //return 0;

//}
//
//
//
//however, what if we wanted to put the output line into its own function 
//as well
//
/*#include <iostream>
int getValueFromUser(){
    std::cout<<"Enter a number: ";
    int input{ };
    std::cin>>input;
    return 0;


}

void printDouble(){
     int num = { getValueFromUser() };
     std::cout<<num<<" doubled is "<<num * 2<<std::endl;
}

int main(){
    printDouble();
    return 0;


}*/

//the above program results:
//Enter a number: 4
//0 doubled is 0
//
//the program still doesn't work correctly(it always print "0 doubled is : 0"), //The core of the problem here is that function  printDouble doesn't have a 
//way to access the value the user entered
// we need some way to pass the value of variable num to function printDouble 
// so that printDouble can use that value in the function body.
//
// function parameters and arguments
//
// In many cases, it is useful to be able to pass information to
// a function to a function being called, so that the function
// has data to work with. for example, if we wanted to write a function
// to add two numbers, we need some way to tell the function which two
// numbers to add when we call it, otherwise, how would the function know what 
// to add? we do that via function parameters and arguments.
//
// A function parameters is a variable used in a function. function parameters
// work almost identically to variables defined inside the function,
// but with one difference:they are always initialized with a value provided by
// the caller of the function.
//
// function parameters are defined in the function declaration by placing them
// in between the parentesis（括号） after the function identifier, 
// with multiple 
// parameters being separated by commas(逗号)．
// here's some example of functions with different numbers of parameters:
// This function takes no parameters
// It doesn't rely on the caller for anything
// void doPrint()
// {
//     std::cout<<"In doPrint() "<<std::endl;
//
// }
//
//This function takes one integer parameter named x
//The caller will supply the value of x
//void printValue(int x)
//{
//    std::cout<<x<<std::endl;
//
//}
//
//This function has two integer parameters, one named x, and one named y
//This caller will supply the value of both x and y
//int add(int x, int y){
//    return x + y;
//
//} 
//
//An argument is a value that is passed from the caller to the function
//when a function call is made:
//doprint(); this call has no parameters
//printValue(6); 6 is the argument passed to function printValue()
//add(2,3); //2 and  3 are the arguments passed to function add()
//how parametes and arguments work together
/*
#include <iostream>
void printValues(int x, int y){
     std::cout<<x<<std::endl;
     std::cout<<y<<std::endl;
     
}

int main(){
    printValues(6,7);
    return 0;

}
*/


// We now have the tool we need to fix the program we presented at the top of
// the lession
# include <iostream>

int getValueFromUser() //this function now return an integer number
{   
    std::cout<<"Enter a number : ";
    int input{};
    std::cin>>input;
    return input;  //added return statement to return input back to the caller

}

void printDouble(int value){
     
     std::cout<<value<<" doubled is "<<value * 2 <<std::endl;


}


int main(){
    //variable num is first initialized with the value entered
    //by the user. then ,function printDouble is called, and
    //the value of argument num is copied into the value parameter of 
    //function printDouble,function printDouble then uses the value of 
    //parameters value
    //int num = { getValueFromUser() };
    //printDouble(num);
    //now we are using the return of function getValueFromUser directly
    //as an argument to function printDouble
    printDouble(getValueFromUser());
    return 0;

}







 
