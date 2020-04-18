//erase() 函数可以删除 string 中的一个子字符串。它的一种原型为：
//string& erase (size_t pos = 0, size_t len = npos);
//
//pos 表示要删除的子字符串的起始下标，len 表示要删除子字符串的长度。如果不指明 len 的话，那么直接删除从 pos 到字符串结束处的所有字符（此时 len = str.length - pos）。

#include <iostream>
#include <string>
using namespace std;

int main(){
    string s1, s2, s3;
    s1 = s2 = s3 = "789456123";
    s2.erase(5, 3);
    s3.erase(6, 2);
    cout<<s1<<endl;
    cout<<s2<<endl;
    cout<<s3<<endl;
    return 0;
}
