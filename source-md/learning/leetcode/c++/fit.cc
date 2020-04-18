#include <iostream>
int fit(int N){
    // base case
    if (N==1 || N==2){
        return 1;
    }
    int curr = 1; int prev = 1;
    for (int i=3; i<=N; i++){
        int sum = curr + prev;
         prev = curr;
         curr = sum;
    }
    return curr;
}

int main() {
    printf("N = 3 , fit = %d\n",fit(4));
    return 0;
    
}