//和普通成员函数一样，构造函数是允许重载的。一个类可以有多个重载的构造函数，
//创建对象时根据传递的实参来判断调用哪一个构造函数。
//
//构造函数的调用是强制性的，一旦在类中定义了构造函数，
//那么创建对象时就一定要调用，不调用是错误的。如果有多个重载的构造函数，
//那么创建对象时提供的
//实参必须和其中的一个构造函数匹配；反过来说，创建对象时只有一个
//构造函数会被调用。
//在C++中，有一种特殊的成员函数，它的名字和类名相同，没有返回值，不需要用户显式调用（用户也不能调用），而是在创建对象时自动执行。这种特殊的成员函数就是构造函数（Constructor）。
//
#include <iostream>
using namespace std;
class Student{
private:
    char *m_name;
    int m_age;
    float m_score;
public:
    //使用重载机制，即两个构造函数名字是一样的
    //这里先声明两个构造函数
    Student();
    Student(char *name, int age, float score);
    //声明普通成员函数
    void setname(char *name);
    void setage(int age);
    void setscore(float score);
    void show();

       
};

//定义类的第一个构造函数
Student::Student(){
    m_name = NULL;
    m_age = 0;
    m_score = 0.0;
}
//定义类的第二个构造函数
Student::Student(char *name, int age, float score){
    m_name = "小华";
    m_age = 29;
    m_score = 99.1f;
}
//定义类的普通成员函数
void Student::setname(char *name){
    m_name = name; 

}
void Student::setage(int age){
    m_age = age;
}
void Student::setscore(float score){
    m_score = score;
}
void Student::show(){
    if (m_name == NULL || m_age<0){
        cout<<"成员变量还未初始化"<<endl;
    }else{
    cout<<m_name<<"的年龄是"<<m_age<<", 成绩是"<<m_score<<endl;
    }
}

int main(){
    //调用构造函数Student(char *name, int age, float score)
    //在栈上创建对象时向构造函数传入参数
    Student stu("小明", 28, 98.4f);
    stu.show();
    //调用构造函数Studen()
    Student *pStu = new Student();
    pStu -> show();
    pStu -> setname("李华");
    pStu -> setage(32);
    pStu -> setscore(96.4f);
    pStu -> show();
    delete pStu;
    return 0;

}

//构造函数Student(char *, int, float)为各个成员变量赋值，
//构造函数Student()将各个成员变量的值设置为空，它们是重载关系。
//根据Student()创建对象时不会赋予成员变量有效值，
//所以还要调用成员函数 setname()、setage()、setscore() 来给它们重新赋值。
//
//构造函数在实际开发中会大量使用，它往往用来做一些初始化工作，
//例如对成员变量赋值、预先打开文件等。
