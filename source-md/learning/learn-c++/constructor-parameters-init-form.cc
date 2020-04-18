//构造函数的一项重要功能是对成员变量进行初始化，
//为了达到这个目的，可以在构造函数的函数体中对成员变量一一赋值，
//还可以采用参数初始化表。
//
#include <iostream>
using namespace std;

class Student{
private:
    char *m_name;
    int m_age;
    float m_score;
public:
    //声明一个构造函数
    Student(char *name, int age, float score);
    void say();

};
//利用参数初始化表来给构造函数的函数体中的成员变量赋值
Student::Student(char *name, int age, float score):m_name(name),m_age(age),m_score(score){
}

void Student::say(){
    cout<<m_name<<"的年龄是"<<m_age<<", 成绩是"<<m_score<<endl;
}

int main(){
    //在堆上创建对象
    Student *pStu = new Student("小明",24, 98.7f);
    pStu -> say();
    delete pStu;
    // 在栈上创建对象
    Student stu("张三", 14, 56);
    stu.say();
    return 0;


}

//如本例所示，定义构造函数时并没有在函数体中对成员变量一一赋值
//其函数体为空（当然也可以有其他语句），而是在函数首部与函数体之间
//添加了一个冒号:，后面紧跟m_name(name), m_age(age), m_score(score)语句，
//这个语句的意思相当于函数体内部的m_name = name; m_age = age; 
//m_score = score;语句，也是赋值的意思。

//使用参数初始化表并没有效率上的优势，仅仅是书写方便，
//尤其是成员变量较多时，这种写法非常简明明了。 
