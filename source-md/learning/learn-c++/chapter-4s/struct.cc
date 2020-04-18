#include <iostream>
#include <string>

//g++ -std=c++11 -fdiagnostics-color=auto --no-warn ~/w2019a/learn-c++/chapter-4s/struct.cc -o ~/w2019a/learn-c++/chapter-4s/struct.o
// declearation a struct
struct Employee{
    short id;
    int age;
    std::string wage;
};
//A big advantage of using structs over individual variables is that we can pass the entire struct to a function that needs to work with the members:
void PrintInformation(Employee employee){

    std::cout<<"ID: "<< employee.id <<std::endl;
    std::cout<<"Age: "<< employee.age <<std::endl;
    std::cout<<"Wage: " << employee.wage << std::endl;
}


// A function can also return a struct , which is one of the 
// few ways to have a function return multiple variable.

struct Point3d{

    double x;
    double y;
    double z;

};

Point3d getZeroPoint(){

    Point3d temp={0.0, 0.0, 0.0};
    return temp;

}


int main(){

    Employee joe = {14, 27,"10k" };
    Employee frank = {15, 25,"12k"};
    // Print joe information
    PrintInformation(joe);
    std::cout<<std::endl;
    // print frank information
    PrintInformation(frank);


    Point3d zero = getZeroPoint();
    if (zero.x== 0.0&&zero.y==0.0&&zero.x==0.0)
        std::cout<<"The point is zero"<<std::endl;
    else
        std::cout<<"The point is not zero"<<std::endl;
    return 0;


}
