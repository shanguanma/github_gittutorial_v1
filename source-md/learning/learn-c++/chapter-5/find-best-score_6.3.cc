//
#include <iostream>
#include <iterator>  // for std::size() // requires c++17

//  g++ -std=c++11 source-md/egs/learn-c++/chapter-5/find-best-score_6.3.cc -o source-md/egs/learn-c++/chapter-5/find-best-score_6.3.o`
int main()
// find best score(e.g. highest score) in one class.
{
    const int scores[] = {89,91,93,97,76}; // assum here they are 5 student's scores
    int max_score = 0;
    int numstudents = sizeof(scores)/sizeof(scores[0]);
    for (int student=0 ;student < numstudents ;++student)
        if(scores[student] > max_score)
            max_score = scores[student];
    std::cout<< "the best score is: "<<max_score<<std::endl;
    return 0;

}
