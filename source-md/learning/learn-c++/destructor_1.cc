#include <iostream>
#include <string>

using namespace std;

class Demo{
public:
    //constructor declaration
    Demo(string s);
    //destructor declaration
    ~Demo();
private:
    string m_s;

};

Demo::Demo(string s): m_s(s){ }
Demo::~Demo(){ cout<<m_s<<endl; }

void func(){
    //把Demo　这个类进行实例化，实例化的对象是obj1,并且传入参数为"1"的字符串
    //因为这个类的对象实例化在函数的内部，所以称为局部对象（local object）
    //并且这个类的对象实例化（创建）是在栈上的，这个创建的对象不需要用后手动删除
    Demo  obj1("1");
}

//global object,把类的实例对象放在函数外边，就称为全局对象
Demo obj2("2");

int main(){
    //local object
    Demo obj3("3");
    //new创建的对象
    Demo *pobj4 = new Demo("4");
    delete pobj4;
    func();
    cout<<"main"<<endl;

    return 0;

}
