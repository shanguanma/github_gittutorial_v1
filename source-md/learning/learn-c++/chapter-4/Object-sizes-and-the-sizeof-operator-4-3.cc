#include <iostream>

int main(){
    std::cout<< "bool:\t\t" <<sizeof(bool)<<" bytes" <<"\t\t"<<sizeof(bool)*8 <<"bits\n";
    std::cout<< "char:\t\t" <<sizeof(char)<<" bytes\n";
    std::cout<< "wchar_t:\t"<<sizeof(wchar_t)<<" bytes\n";
    std::cout<< "char16_t:\t"<<sizeof(char16_t) << " bytes\n";//c++11 only
    std::cout<< "char32_t:\t"<<sizeof(char32_t) << " bytes\n";//c++11 only
    std::cout<< "short:\t\t"<<sizeof(short)<<" bytes\n";
    std::cout<< "int:\t\t"<<sizeof(int)<<" bytes\n";
    std::cout<< "long:\t\t"<<sizeof(long)<< " bytes\n";
    std::cout<< "long long:\t"<<sizeof(long long)<< " bytes\n";//c++11 only
    std::cout<< "float:\t\t"<<sizeof(float)<< " bytes\n";
    std::cout<< "double:\t\t" <<sizeof(double)<< " bytes\n";
    std::cout<< "long double:\t"<<sizeof(long double)<< " bytes\n";


    // opterate a varible
    int x{};
    std::cout<< "x is "<< sizeof(x)<<" bytes"<<std::endl;
     
    return 0;

}
//results:
//bool:		1 bytes
//char:		1 bytes
//wchar_t:	4 bytes
//char16_t:	2 bytes
//char32_t:	4 bytes
//short:		2 bytes
//int:		4 bytes
//long:		8 bytes
//long long:	8 bytes
//float:		4 bytes
//double:		8 bytes
//long double:	16 bytes
//x is 4 bytes
//
