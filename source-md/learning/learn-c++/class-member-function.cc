#include <iostream>

using namespace std;

//方法一：member function 在类体中定义成员函数
//class Student{
//
//public:
    //member variable
//    char *name;
//    int age;
//    float score;
    //member function
    
//    void say(){
//        cout<<name<<"的年龄是"<<age<<", 成绩是"<<score<<endl;
//    }
//};

//方法二：在类体中使用函数声明，而将函数定义在类体外面
class Student{
public:
    char *name;
    int age;
    float score;

    void say(); //function declaration
 };
//fuuction define
// 但当成员函数定义在类外时，就必须在函数名前面加上类名予以限定。
// ::被称为域解析符（也称作用域运算符或作用域限定符），
// 用来连接类名和函数名，指明当前函数属于哪个类。
//
// 成员函数必须先在类体中作原型声明，然后在类外定义，
// 也就是说类体的位置应在函数定义之前。
 void Student :: say(){
     cout<<name<<"的年龄是"<<age<<",成绩是"<<score<<endl;
}

//inline 成员函数
//在类体中和类体外定义成员函数是有区别的：在类体中定义的成员函数会自动成
//为内联函数，在类体外定义的不会。当然，在类体内部定义的函数也可以加 
//inline 关键字，但这是多余的，因为类体内部定义的函数默认就是内联函数。
//
//内联函数一般不是我们所期望的，它会将函数调用处用函数体替代，
//所以我建议在类体内部对成员函数作声明，而在类体外部进行定义，
//这是一种良好的编程习惯，实际开发中大家也是这样做的。
//
//当然，如果你的函数比较短小，希望定义为内联函数，那也没有什么不妥的。
//


//这里在堆上创建类的对象
int main(){

    Student *pStu = new Student;
    pStu -> name = "小明";
    pStu -> age = 29;
    pStu -> score = 98.3f;
    pStu -> say();
    delete pStu;
    
    return 0;
 

}

