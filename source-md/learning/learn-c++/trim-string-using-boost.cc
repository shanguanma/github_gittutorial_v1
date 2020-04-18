#include <string>
#include <iostream>
#include <boost/algorithm/string.hpp>

using namespace std;

//Trim string on original string ,so original string is changed ,default trim char (white space)
void example_trim(std::string & msg){
    
     std::cout<<"Original string is ,"<<msg<<endl;
     boost::algorithm::trim(msg);
     std::cout<<"trim both side,"<<"["<<msg<<"]"<<endl;
     boost::algorithm::trim_right(msg);
     std::cout<<"trim right side"<<"["<<msg<<"]"<<endl;
     boost::algorithm::trim_left(msg);
     std::cout<<"trim left side"<<"["<<msg<<"]"<<endl;
    
}


//void example_trim_copy(std::string &msg, std::string & newStr){
//
//     std::cout<<"Original string is ,"<<msg<<endl;
//
//     std::string newStr = boost::algorithm::trim_copy(msg);     
//     std::cout<<"trim both side,"<<"["<<newStr<<"]"<<endl;
//
//     std::string newStr = boost::algorithm::trim_right_copy(msg);
//     std::cout<<"trim right side"<<"["<<newStr<<"]"<<endl;
//        
//     std::string newStr = boost::algorithm::trim_left_copy(msg);
//     std::cout<<"trim left side"<<"["<<newStr<<"]"<<endl;
//
//
//}

// coustom trim char i.e trim char is colon.

//bool isColon(char c){
//
//     return c = ':';

//}

//void example_trim_if(std::string &msg){
  
//     boost::algorithm::trim_if(msg, &isColon);
//     std::cout<<"trim both side,"<<"["<<msg<<"]"<<endl;
//     boost::algorithm::trim_right_if(msg, &isColon);
//     std::cout<<"trim right side"<<"["<<msg<<"]"<<endl;
//     
//     boost::algorithm::trim_left_if(msg, &isColon);
//     std::cout<<"trim left side"<<"["<<msg<<"]"<<endl;

//}

int main(){
    std::string msg = "::StartMsg :: Hello :: EndMsg::";
    //bool isColon(char c){return c = ':';}
    //boost::algorithm::trim_if(msg, ':');
    //std::cout<<"trim both side,"<<"["<<msg<<"]"<<endl;
    //boost::algorithm::trim_right_if(msg, &isColon);
    //std::cout<<"trim right side"<<"["<<msg<<"]"<<endl;

    //boost::algorithm::trim_left_if(msg, &isColon);
    //std::cout<<"trim left side"<<"["<<msg<<"]"<<endl;

    //example_trim_copy(msg, newStr);


    std::string str = "  Start :: Hello :: End   ";
    example_trim(str);
    return 0;

}
