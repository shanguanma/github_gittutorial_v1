//提取子字符串
//substr() 函数用于从 string 字符串中提取子字符串，它的原型为：
//string substr (size_t pos = 0, size_t len = npos) const;
//
//pos 为要提取的子字符串的起始下标，len 为要提取的子字符串的长度。
//

#include <iostream>
#include <string>

using namespace std;

int main(){
    string s1 = "first second third";
    string s2;
    s2 = s1.substr(6, 6);
    cout<<s1<<endl;
    cout<<s2<<endl;
    return 0;

}
