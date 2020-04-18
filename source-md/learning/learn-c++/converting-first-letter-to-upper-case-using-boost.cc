#include <iostream> //std::cout,std::cin
#include <string>
#include <vector>  //std::vector, std::begin, std::end
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/iterator/filter_iterator.hpp>

using namespace std;

struct IsFirstLetter{
    //return true if last char was ' ' &current char is an alphabet
    bool operator()(char & c){
        static int last = ' ';
        if(last == ' ' && c != ' ' && ::isalpha(c)){
            last = c;
            return true;
         }
         last = c;
         return false;
    }
};

int main(){
    std::string data = "hi this is a sample string";
    typedef boost::filter_iterator<IsFirstLetter, std::string::iterator> FilterIter;

    //creating filter i.e. a function object
    IsFirstLetter isFirstLetterObj;
    //
    //creat start of Filtered Iterator
    FilterIter filter_iter_first(isFirstLetterObj, data.begin(),data.end());
    
    //creat end of Filtered Iterator
    //
    FilterIter filter_iter_last(isFirstLetterObj, data.end(), data.end());
    
    //creat a range from start & end filtered iterator object
    boost::iterator_range<FilterIter> strRange;
    strRange = boost::make_iterator_range(filter_iter_first, filter_iter_last);
    //convert string to upper case
    boost::to_upper(strRange);
    
    std::cout<<data<<std::endl;
    return 0;


}
