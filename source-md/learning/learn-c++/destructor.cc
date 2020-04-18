//创建对象时系统会自动调用构造函数进行初始化工作，同样，
//销毁对象时系统也会自动调用一个函数来进行清理工作，
//例如释放分配的内存、关闭打开的文件等，这个函数就是析构函数。
//
//析构函数（Destructor）也是一种特殊的成员函数，没有返回值，
//不需要程序员显式调用（程序员也没法显式调用），而是在销毁对象时自动执行。
//构造函数的名字和类名相同，而析构函数的名字是在类名前面加一个~符号。
//
//注意：析构函数没有参数，不能被重载，因此一个类只能有一个析构函数。
//如果用户没有定义，编译器会自动生成一个默认的析构函数。
#include <iostream>
using namespace std;

class VLA{
public:
    VLA(int len); //constructor
    ~VLA(); //destructor
public:
    void input(); //Input array element from the console
    void show();  //display array element
private:
    int *at(int i); //get the pointor to the ith element
private:
    const int m_len; //array length
    int *m_arr; //array pointor
    int *m_p; // 指向数组第ｉ个元素的指针

};



VLA::VLA(int len): m_len(len){
    if (len > 0){ m_arr = new int[len]; /*分配内存*/}
    else{ m_arr = NULL;}
}
VLA::~VLA(){
    delete[] m_arr; //释放内存
}

//define class normal member funtion
void VLA::input(){
    for(int i=0; m_p=at(i); i++){ cin>>*at(i);}
}

void VLA::show(){
    for(int i=0; m_p=at(i); i++){
        if(i == m_len - 1){ cout<<*at(i)<<endl;}
        else{ cout<<*at(i)<<", ";}
    }
}

int * VLA::at(int i){
    if(!m_arr || i<0 || i>=m_len){ return NULL;}
    else{ return m_arr + i;}
}

int main(){
    //创建一个有ｎ个元素的数组（对象）
    int n;
    cout<<"Input araay length: ";
    cin>>n;
    VLA *parr = new VLA(n);
    //input array element
    cout<<"Input "<<n<<" numbers: "<<endl;
    parr -> input();
    //output array element
    cout<<"Elements: "<<endl;
    parr -> show();
    //delete array (object)
    delete parr;
    return 0;

}
