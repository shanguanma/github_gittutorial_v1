//在C++中，有一种特殊的成员函数，它的名字和类名相同，没有返回值，
//不需要用户显式调用（用户也不能调用），而是在创建对象时自动执行。
//这种特殊的成员函数就是构造函数（Constructor）。
#include <iostream>
using namespace std;


class Student{
private:
    char *m_name;
    int m_age;
    float m_score;
public:
    //declaration constructor
    Student(char *name, int age, float score);
    //declaration normal member function
    void say();
    

};


//define constructor 
Student::Student(char *name, int age, float score){
    m_name = name;
    m_age = age;
    m_score = score;
}
//define normal member function
void Student::say(){
    std::cout<<m_name<<"的年龄是"<<m_age<<", 成绩是"<<m_score<<endl;

}

int main(){

    //passes arguments to the constructor when an object 
    //is created on the stack
    Student stu("小华",29,98.3f);
    stu.say();
    //passes argument to the constructor when an object 
    //is created on the heap(堆)
    Student *pStu = new Student("小张",31, 99.1f);
    pStu -> say();
    
    delete pStu;
    return 0;
}
