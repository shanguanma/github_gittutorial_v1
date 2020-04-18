#include <vector>
#include <iostream>

//this is dynamic programming problem
class Solution{
public:
    int climbing_stairs(int n){
        std::vector<int> dp(n + 3, 0);
        dp[1] = 1;
        dp[2] = 2;
        for (int i = 3; i<=n; i++){
            dp[i] = dp[i-1] + dp[i-2];
        }
        return dp[n];

    }


};

int main(){
    Solution solve;
    printf("%d\n", solve.climbing_stairs(4));
    return 0;
}
