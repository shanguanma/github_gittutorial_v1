"""
Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
Examples:
s = "leetcode"
return 0.
s = "loveleetcode",
return 2.
Note: You may assume the string contain only lowercase letters.
给定一个字符串，找到第一个不重复的字符，输出索引，如不存在输出 -1。
思路是直接使用字典，都是O(1)。

测试地址：
https://leetcode.com/problems/first-unique-character-in-a-string/description/
"""

# method 1
class Solution(object):
    def firstUniqChar(self, s):
        """
        :type: str
        :rtype: int
        """
        chars = set(s)
        print('chars:', chars) 
        print("type of chars:", type(chars))
        # why using min function ,
        # Beacuse The Title requires finding the index number of the first non-repetitive letter. 
        # There may be more than one non-repetitive letter, but for the first time, of course, the one with the smallest index number fits the title. 
        # method 1
        #return min([s.index(char) for char in chars if s.count(char) == 1] or [-1])
        # method 2
        #return min([s.index(char) for char in chars if s.count(char) == 1], default=-1)
        # method 3
        index = [s.index(char) for char in chars if s.count(char) == 1]
        return min(index) if len(index)>1 else -1



a = Solution()
s = "loveleetcode"
print(a.firstUniqChar(s))
