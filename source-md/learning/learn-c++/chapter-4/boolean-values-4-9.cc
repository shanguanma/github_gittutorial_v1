#include <iostream>
//g++ -std=c++11  -fdiagnostics-color=auto -W chapter-4/boolean-values-4-9.cc -o chapter-4/boolean-values-4-9.o
//-W 取消warning的显示
//-fdiagnostics-color=auto 错误高亮显示
//-std=c++11　使用c++11 标准命名空间
bool isEqual(int x, int y){

    return(x == y);
}


int main(){

    std::cout<< "you input an integer: ";
    int x{0};
    std::cin>>x;
    std::cout<< "you input an other integer: ";
    int y{0};
    std::cin>>y;
    std::cout<<std::boolalpha;
    std::cout<< x << " and "<< y << " equal ? ";
    std::cout<< isEqual(x, y)<<std::endl;
    return 0; 
    

}
