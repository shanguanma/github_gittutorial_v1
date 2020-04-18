//C++通过 public、protected、private 三个关键字来控制成员变量和
//成员函数的访问权限，它们分别表示公有的、受保护的、私有的，
//被称为成员访问限定符。所谓访问权限，就是你能不能使用该类中的成员。
//Java、C# 程序员注意，C++ 中的 public、private、protected 
//只能修饰类的成员，不能修饰类，C++中的类没有共有私有之分。
//在类的内部（定义类的代码内部），无论成员被声明为 public、protected 
//还是 private，都是可以互相访问的，没有访问权限的限制。
//
//在类的外部（定义类的代码之外），只能通过对象访问成员，
//并且通过对象只能访问 public 属性的成员，
//不能访问 private、protected 属性的成员
#include <iostream>
//#pragma warning
using namespace std;

//class declaraioin
class Student{
private:
    char *m_name;
    //const char name[];
    int m_age;
    float m_score;
public:
    void setname(char *name);
    void setage(int age);
    void setscore(float score);
    void show();

        
};

//member function define
void Student :: setname(char *name){
    m_name = name;
}
void Student :: setage(int age){
    m_age = age;
}
void Student :: setscore(float score){
    m_score = score;
}
void Student :: show(){
    std::cout<<m_name<<"的年龄是"<<m_age<<",成绩是"<<m_score<<endl;
}



int main(){
    //在栈上创建类的对象,
    //不用删除对象， 程序自动管理内存
    Student stu;
    stu.setname("小明");
    stu.setage(15);
    stu.setscore(98.3f);
    stu.show();
   
   
    
    //在堆上创建类的对象，
    //必须指定类的对象指针，
    //而且必须使用后删除对象，防止内存垃圾堆积
    Student *pStu = new Student;
    pStu -> setname("小华");
    pStu -> setage(29);
    pStu -> setscore(98);
    pStu -> show();
    //delete pStu; //delete object
    
    return 0;

}   


