//this 是 C++ 中的一个关键字，也是一个 const 指针，它指向当前对象，
//通过它可以访问当前对象的所有成员。
//
//所谓当前对象，是指正在使用的对象(这里的对象就是类的实例化)。
//例如对于stu.show();，
//stu 就是当前对象，this 就指向 stu。
#include <iostream>

using namespace std;
class Student{
public:
    //member function declaration
    void setname(char *name);
    void setage(int age);
    void setscore(float score);
    void show();
    void printThis();
private:
    //char *m_name;
    //int m_age;
    //float m_score;
    char *name;
    int age;
    float score;

};

//his 只能用在类的内部，通过 this 可以访问类的所有成员，
//包括 private、protected、public 属性的。
//本例中成员函数的参数和成员变量重名，只能通过 this 区分。
//以成员函数setname(char *name)为例，它的形参是name，
//和成员变量name重名，如果写作name = name;这样的语句，
//就是给形参name赋值，而不是给成员变量name赋值。
//而写作this -> name = name;后，=左边的name就是成员变量，
//右边的name就是形参，一目了然。
//
//注意，this 是一个指针，要用->来访问成员变量或成员函数。
//
//this 虽然用在类的内部，但是只有在对象被创建以后才会给 this 赋值，
//并且这个赋值的过程是编译器自动完成的，不需要用户干预，
//用户也不能显式地给 this 赋值。本例中，this 的值和 pstu 的值是相同的。

void Student::setname(char *name){
    //this -> m_name = name;
    this->name = name;
}
void Student::setage(int age){
    //this -> m_age = age;
    this->age = age;
}
void Student::setscore(float score){
    //this -> m_score = score;
    this->score = score;
}

void Student::show(){
    //cout<<this -> m_name<<"的年龄是"<<this -> m_age<<", 成绩是"<<this -> m_score<<endl;
    cout<<this->name<<"的年龄是"<<this->age<<", 成绩是"<<this->score<<endl;
}

//我们不妨来证明一下，给 Student 类添加一个成员函数printThis()，
//专门用来输出 this 的值，如下所示：
void Student::printThis(){
    cout<<this<<endl;
}
int main(){
    Student *pstu = new Student;
    pstu -> setname("小华");
    pstu -> setage(29);
    pstu -> setscore(98.4f);
    pstu -> show();
    delete pstu;
    
    Student *pstu1 = new Student;
    pstu1 -> printThis();
    cout<<pstu1<<endl;
    delete pstu1;
    
    Student *pstu2 = new Student;
    pstu2 -> printThis();
    cout<<pstu2<<endl;
    delete pstu2;
    
    return 0;


}
