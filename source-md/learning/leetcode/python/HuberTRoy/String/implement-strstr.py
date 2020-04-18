"""
Implement strStr().
Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.
Example 1:
Input: haystack = "hello", needle = "ll"
Output: 2
Example 2:
Input: haystack = "aaaaa", needle = "bba"
Output: -1
Clarification:
What should we return when needle is an empty string? This is a great question to ask during an interview.
For the purpose of this problem, we will return 0 when needle is an empty string. This is consistent to C's strstr() and Java's indexOf().
strStr() 等同于 Python 中的 find()
想要通过直接：
```
str.find()
```
直接 beat 100%...
测试用例：
https://leetcode.com/problems/implement-strstr/description/
自己的实现思路：
记录两者len，needle的长度大于haystack，则一定不存在 -1.
接下来遍历 haystack 若此时剩余长度小于 needle 则一定不存在 -1.
若[0]与[0]相同，则进一步进行对比，一样则返回下标，不一样则继续。
"""
# python3 code style
class Solution(object):
    def strStr(self, haystack: 'str', needle: 'str') -> 'int':
        # method 1
        #return haystack.find(needle)
        # method 2
        if not needle:
            return 0
        lengthhaystack = len(haystack)
        lengthneedle = len(needle)
        if lengthhaystack < lengthneedle:
            return -1

        if lengthhaystack == lengthneedle:
            return 0 if haystack == needle else -1
        for i, d in enumerate(haystack):
            # 此时剩余长度小于lengthneedle ,则一定不存在，返回-1
            if lengthhaystack - i < lengthneedle:
                return -1
            if d == needle[0]:
                if haystack[i:i+lengthneedle] == needle:
                    return i



        

haystack = "hello"
needle = "ll"

a = Solution()
print(a.strStr(haystack, needle))
        
