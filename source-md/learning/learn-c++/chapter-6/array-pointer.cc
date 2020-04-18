#include <iostream>

int main(){
    int array[5]= {1,3,4,5,6};
    std::cout<<"element 0 has address: "<<&array[0]<<std::endl;
    std::cout<<"the array decays to the pointer holding address: "<<array<<std::endl;
    std::cout<<"get the array address  via &an array: "<<&array<<std::endl;
    std::cout<<"the first element via derefrencing an array: "<<*array<<std::endl;

    char name[] = "Jason";
    std::cout<<"the first letter is : "<<*name<<std::endl;


    char *ptr = name;
    std::cout<<"get first letter of the string  via an pointer: "<<*ptr<<std::endl;

    std::cout<<sizeof(name)<<std::endl;
    std::cout<<sizeof(array)<<std::endl;
    std::cout<<"the pointer of the name : "<<sizeof(*name)<<std::endl;
    std::cout<<"the pointer lenght of the array: "<<sizeof(*array)<<std::endl;


    return 0;




}
