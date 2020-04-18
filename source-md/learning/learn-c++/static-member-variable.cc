//对象的内存中包含了成员变量，不同的对象占用不同的内存，
//这使得不同对象的成员变量相互独立，它们的值不受其他对象的影响。
//例如有两个相同类型的对象 a、b，它们都有一个成员变量 m_name，
//那么修改 a.m_name 的值不会影响 b.m_name 的值。
//
//可是有时候我们希望在多个对象之间共享数据，对象 a 改变了某份数据后
//对象 b 可以检测到。共享数据的典型使用场景是计数，
//以前面的 Student 类为例，如果我们想知道班级中共有多少名学生，
//就可以设置一份共享的变量，每次创建对象时让该变量加 1。
//
//在C++中，我们可以使用静态成员变量来实现多个对象共享数据的目标。
//静态成员变量是一种特殊的成员变量，它被关键字static修饰.
//
#include <iostream>

using namespace std;

class Student{
public:
    //delaration constructor
    Student(char *name, int age, float score);
    //delaration member function
    void show();
public:
    static int m_total;
private:
    char *m_name;
    int m_age;
    float m_score;

};

//Initialize static member variable
int Student::m_total = 0;
//使用构造函数方便给类的成员变量赋值，并且使用参数初始化表进一步方便书写
Student::Student(char *name, int age, float score): m_name(name), m_age(age),m_score(score){
    m_total++;  //operate static member vaiable

}

void Student::show(){
    cout<<m_name<<"的年龄是"<<m_age<<", 成绩是"<<m_score<<" (当前共有"<<m_total<<"名学生)"<<endl;
}

int main(){
    //creat anonymous objects
    //之所以使用匿名对象，是因为每次创建对象后只会使用它的 show() 函数，
    //不再进行其他操作。不过使用匿名对象无法回收内存，会导致内存泄露
    //(new Student("小明", 15, 96.3f)) -> show();
    //(new Student("李磊", 16, 97.3f)) -> show();
    //(new Student("张华", 16, 90)) -> show();
    //(new Student("张三", 14, 99.6f)) -> show();
    Student *pstu1 = new Student("小明", 15, 96.3f);
    pstu1 -> show();
    delete pstu1;
    Student *pstu2 = new Student("李磊", 16, 97.3f);
    pstu2 -> show();
    delete pstu2;
    Student *pstu3 = new Student("张华", 16, 90);
    pstu3 -> show();
    delete pstu3;
    Student *pstu4 = new Student("张三", 14, 99.6f);
    pstu4 -> show();
    delete pstu4;
    
    return 0;
}


//强调一下:
//1) 一个类中可以有一个或多个静态成员变量，所有的对象都共享这些静态成员变量，
//都可以引用它。
//
//2) static 成员变量和普通 static 变量一样，都在内存分区中的全局数据区分配内存,
//到程序结束时才释放。这就意味着，static 成员变量不随对象的创建而分配内存，
//也不随对象的销毁而释放内存。而普通成员变量在对象创建时分配内存，
//在对象销毁时释放内存。
//
//3) 静态成员变量必须初始化，而且只能在类体外进行。例如：
//int Student::m_total = 10;
//
//初始化时可以赋初值，也可以不赋值。如果不赋值，那么会被默认初始化为 0。
//全局数据区的变量都有默认的初始值 0，而动态数据区（堆区、栈区）
//变量的默认值是不确定的，一般认为是垃圾值。
//
//4) 静态成员变量既可以通过对象名访问，也可以通过类名访问，
//但要遵循 private、protected 和 public 关键字的访问权限限制。
//当通过对象名访问时，对于不同的对象，访问的是同一份内存。
