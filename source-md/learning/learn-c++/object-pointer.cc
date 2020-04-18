#include <iostream>

using namespace std;

class Student{
public:
    char *name;
    int age;
    float score;
    void say(){
        cout<<name<<"的年龄是"<<age<<",成绩是"<<score<<endl;
    }
};


int main(){
    //当然，你也可以在堆上创建对象，这个时候就需要使用前面讲到的new关键字
    //*pStu is 对象指针，并且是在堆上创建的对象，
    //使用 new 在堆上创建出来的对象是匿名的，没法直接使用，
    //必须要用一个指针指向它，再借助指针来访问它的成员变量或成员函数。
    //堆内存由程序员管理，对象使用完毕后可以通过 delete 删除。
    //在实际开发中，new 和 delete 往往成对出现，以保证及时删除不再使用的对象，
    //防止无用内存堆积。
    //有了对象指针后，可以通过箭头->来访问对象的成员变量和成员函数，
    //这和通过结构体指针来访问它的成员类似。
    //
    //下面这个是在栈上创建的对象
    //Student stu;
    //Student *pStu = &stu; //pStu 是一个指针， 它指向Student类型的数据， 
    //也就是通过Student创建出来的对象，
    //栈内存是程序自动管理的，不能使用 delete 删除在栈上创建的对象；
    Student *pStu = new Student;
    pStu -> name = "小明";
    pStu -> age = 29;
    pStu -> score = 98.3f;
    pStu -> say();
    delete pStu; //delete object
    return 0;

}


//下面这个是在栈上创建的对象
// Student stu;
// Student *pStu = &stu; //pStu 是一个指针， 它指向Student类型的数据， 
// 也就是通过Student创建出来的对象，
// 栈内存是程序自动管理的，不能使用 delete 删除在栈上创建的对象；
//
// 完整例子如下：
// #include <iostream>
// using namespace std;
//
// class Student{
// public:
//     char *name;
//     int age ;
//     float score;
//
//     void say(){
//         cout<<name<<"的年龄是"<<age<<",成绩是"<<score<<endl;
//     }
//
// };
//
// int main(){
// 
//     Student stu;
//     stu.name = "小明";
//     stu.age = 29;
//     stu.score = 98.3f;
//     stu.say();
//     return 0;
// }
//
// 总结：本例讲解了两种创建对象的方式：一种是在栈上创建，形式和定义普通变量类似；另外一种是在堆上创建，必须要用一个指针指向它，读者要记得 delete 掉不再使用的对象。
//
// 通过对象名字访问成员使用点号.，通过对象指针访问成员使用箭头->
