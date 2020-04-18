#include <iostream>
using namespace std;


//calss is usually defined outside functions
//
class Student{
public:
    //class contains variables
    char *name;
    int age;
    float score;
    //class contains functions
    void say(){
        cout<<name<<"的年龄是"<<age<<", 成绩是"<<score<<endl;

    } 

};

int main(){

    //creat object
    Student stu;
    stu.name = "小明";
    stu.age = 29;
    stu.score = 99.8f;
    stu.say();
    return 0;
}
