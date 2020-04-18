//在类中，static 除了可以声明静态成员变量，还可以声明静态成员函数。
//普通成员函数可以访问所有成员（包括成员变量和成员函数），
//静态成员函数只能访问静态成员。
//
//编译器在编译一个普通成员函数时，会隐式地增加一个形参 this，
//并把当前对象的地址赋值给 this，所以普通成员函数只能在创建对象后通过对象
//来调用，因为它需要当前对象的地址。而静态成员函数可以通过类来直接调用，
//编译器不会为它增加形参 this，它不需要当前对象的地址，
//所以不管有没有创建对象，都可以调用静态成员函数。
//
//普通成员变量占用对象的内存，静态成员函数没有 this 指针，
//不知道指向哪个对象，无法访问对象的成员变量，也就是说静态成员函数不能访问
//普通成员变量，只能访问静态成员变量。
//
//普通成员函数必须通过对象才能调用，而静态成员函数没有 this 指针，
//无法在函数体内部访问某个对象，所以不能调用普通成员函数，只能调用静态成员函数。
//
//静态成员函数与普通成员函数的根本区别在于：普通成员函数有 this 指针，
//可以访问类中的任意成员；而静态成员函数没有 this 指针，
//只能访问静态成员（包括静态成员变量和静态成员函数）。
//
#include <iostream>

using namespace std;

class Student{
public:
    //declaration constructor
    Student(char *name, int age, float score);
    //declaration member function
    void show();
public:
    //declaration static member function
    static int getTotal();
    static float getPoints();
private:
    static int m_total; //total people
    static float m_points; //total scores

private:
    char *m_name;
    int m_age;
    float m_score;
};

//initailize static vaiable
int Student::m_total = 0;
float Student::m_points = 0.0;

//define constructor
Student::Student(char *name, int age, float score): m_name(name), m_age(age), m_score(score){
    m_total++; //operate static member variable
    m_points +=score; //operate static member variable
}

//define member function
void Student::show(){
    cout<<m_name<<"的年龄是"<<m_age<<", 成绩是"<<m_score<<","<<endl;
}
//static member function 在定义时不能加static
//define static member function
int Student::getTotal(){
    //static member function can call static member variable
    return m_total;
}

float Student::getPoints(){
    //static member function can call static member variable
    return m_points;
}

int main(){
    //using anonymous functions, 
    //Just because it is convenient to write, but can not recover memory, easy to leak memory, 
    //large projects are not recommended to use, so write two ways to achieve, 
    //the second is to create objects on the heap.
    //
    //anonymous functions way:
    //(new Student("小明", 13, 87.4f)) -> show();
    //(new Student("小海", 30, 99.9f)) -> show();
    //(new Student("小猪", 28, 98.6f)) -> show();
    //(new Student("小华", 35, 95.3f)) -> show();
    //create objects on the heap
    Student *pstu1 = new Student("小明", 13, 87.4f);
    pstu1 -> show();
    delete pstu1; //delete object
    Student *pstu2 = new Student("小海", 30, 99.9f);
    pstu2 -> show();
    delete pstu2; //delete object
    Student *pstu3 = new Student("小猪", 28, 98.6f);
    pstu3 -> show();
    delete pstu3; //delete object
    Student *pstu4 = new Student("小华", 35, 95.3f);
    pstu4 -> show(); //delete object
    delete pstu4;
    // 静态成员函数通过类来调用
    int total = Student::getTotal();
    float points = Student::getPoints(); 
    cout<<"当前共有"<<total<<"学生"<<", 总成绩是"<<points<<"，平均分"<<points/total<<endl;
    
    return 0;
    
}


