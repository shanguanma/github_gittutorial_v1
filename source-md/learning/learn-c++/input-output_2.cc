//C语言并没有彻底从语法上支持“真”和“假”，
//只是用 1 和 0 来代表。这点在 C++ 中得到了改善，
//C++ 新增了bool类型，它一般占用 1 个字节长度。
//bool 类型只有两个取值，true 和 false：true 表示“真”，
//false 表示“假”。请看下面的例子：
//遗憾的是，在 C++ 中使用 cout 输出 bool 变量的值时还是用数字 1 和 0 表示，
//而不是 true 或 false。Java、PHP、JavaScript 等也都支持布尔类型，
//但输出结果为 true 或 false
#include <iostream>
using namespace std;

int main(){
    int a, b;
    bool flag; // bool variable
    cin>>a>>b;
    flag = a > b;
    cout<< "flag = "<<flag<<endl;
    return 0;

}
