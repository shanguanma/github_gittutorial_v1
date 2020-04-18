//在类中，如果你不希望某些数据被修改，可以使用const关键字加以限定。
//const 可以用来修饰成员变量、成员函数以及对象。
//const成员变量
//const 成员变量的用法和普通 const 变量的用法相似，
//只需要在声明时加上 const 关键字。初始化 const 成员变量只有一种方法，
//就是通过参数初始化表.
//const成员函数
//const 成员函数可以使用类中的所有成员变量，但是不能修改它们的值，
//这种措施主要还是为了保护数据而设置的。const 成员函数也称为常成员函数。
//
//常成员函数需要在声明和定义的时候在函数头部的结尾加上 const 关键字
//
//g++ -fdiagnostics-color=auto class-const.cc -o class-const.o
//-fdiagnostics-color=auto is 支持诊断信息支持彩色显示
#include <iostream>

using namespace std;

class Student{
public:
    //declaration constructor
    Student(char *name, int age, float score);
    void show();
public:
    //declatation constant function
    char *getname() const;
    int getage() const;
    float getscore() const;

private:
    char *m_name ;
    int m_age;
    float m_score;

};

//define constructor using parameter initialize form

Student::Student(char *name, int age, float score): m_name(name), m_age(age), m_score(score){};

//define member function
void Student::show(){
    cout<<m_name<<"的年龄是"<<m_age<<", 成绩是"<<m_score<<endl;
}

//define constant member function
char * Student::getname() const{
    //can operate private member variable
    return m_name;
}

int Student::getage() const{
    //can operate private member variable
    return m_age;
}
float Student::getscore() const{
    //can operate private member variable
    return m_score;
}


int main(){
    // define constant object on the stack,
    const Student stu("小明", 27, 87.6f);
    //一旦将对象定义为常对象之后，不管是哪种形式，该对象就只能访问被 const 修饰的成员了
    //（包括 const 成员变量和 const 成员函数），因为非 const 成员可能会修改对象的数据
    //（编译器也会这样假设），C++禁止这样做。
    //stu.show();  //error
    //mm_name = stu.getname() //error 
    //mm_age = stu.getage() //error
    //mm_score = stu.getscore() //error
    cout<<stu.getname()<<"的年龄是"<<stu.getage()<<", 成绩是"<<stu.getscore()<<endl;
    
    //define constant object point on the head.
    const Student *pstu = new Student("小华", 34, 98.7f);
    //一旦将对象定义为常对象之后，不管是哪种形式，该对象就只能访问被 const 修饰的成员了
    //（包括 const 成员变量和 const 成员函数），因为非 const 成员可能会修改对象的数据
    //（编译器也会这样假设），C++禁止这样做。
    //pstu -> show(); //error
    cout<<pstu->getname()<<"的年龄是"<<pstu->getage()<<", 成绩是"<<pstu->getscore()<<endl;
    
    delete pstu;
    return 0;
}   
