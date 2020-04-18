# 2017 Haihua Xu 
# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import os
import warnings
import sys
import copy
# reload(sys)
# sys.setdefaultencoding('utf8')
# import unicodedata
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler)

def IsRealNumber(s):
    try:
        val = float(s)
    except ValueError:
        return False
    return True
def IsInteger(s):
    try:
        val = int(s)
    except ValueError:
        return False
    return True
def IsCurrencyNumber(word):
    if not word: return False
    if re.match(r'^\$\d+', word): return True
    if re.match(r'^￥１|２|３|４|５|６|７|８|９|０|[0-9]', word): return True
    return False
def NormalizeNumber(word):
    if IsRealNumber(word): return word
    word0 = word
    word = unicodedata.normalize('NFKC', word)
    if IsRealNumber(word): return word
    return word0
def IsOovNumber(num):
    numDict = {'0':'零', '1':'一', '2':'二', '3':'三','4':'四', '5':'五', '6':'六','7':'七','8':'八','9':'九'}
    numList = list(str(num))
    return __IsOovNumber(numList, numDict)
def __IsOovNumber(numList, numDict):
    for num in numList:
        if not str(num) in numDict:
            return True
    return False
def ReadPositiveInChinese(num):
    if not (IsInteger(num) and  int(num) >=0): return num
    numList = list(str(num))
    nBit = len(numList)
    verbatim = str()
    numDict = {'0':'零', '1':'一', '2':'二', '3':'三','4':'四', '5':'五', '6':'六','7':'七','8':'八','9':'九'}
    if __IsOovNumber(numList, numDict):
        return num
    if re.match(r'^0', str(num)):
        retstr = str()
        for num in numList:
            retstr += numDict[num] + ' '
        return retstr.strip()
    if nBit == 1:
        return  numDict[numList[0]]
    if nBit == 2:
        if numList[0] == '0':
            return  numDict[numList[0]] + ' ' + numDict[numList[1]]
        if numList[1] == '0':
            if numDict[numList[0]] == '一':
                return  ' 十'
            else:
                return numDict[numList[0]] + ' 十'
        if numDict[numList[0]] == '一':
            return ' 十 ' +  numDict[numList[1]]
        else:
            return numDict[numList[0]] + ' 十 ' +  numDict[numList[1]]
    if nBit == 3:
        remainder = int(num) % 100
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder) 
        if remainder == 0:
            return numDict[numList[0]] + ' 百'
        if remainder < 10:
            return numDict[numList[0]] + ' 百 零 ' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 百 ' + ReadPositiveInChinese(remainder)
    if nBit == 4:
        remainder = int(num) % 1000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder) 
        if remainder == 0:
            return numDict[numList[0]] + ' 千'
        if  remainder < 100:
            return numDict[numList[0]] + ' 千 零 ' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 千 ' + ReadPositiveInChinese(remainder)
    if nBit == 5:
        remainder = int(num) % 10000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder) 
        if remainder == 0:
            return numDict[numList[0]] + ' 万'
        if remainder < 1000:
            return numDict[numList[0]] + ' 万 零　' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 万 ' + ReadPositiveInChinese(remainder)
    if nBit == 6:
        remainder = int(num) % 100000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder)
        if remainder == 0:
            return numDict[numList[0]] + ' 十　万'
        if remainder < 10000:
            return numDict[numList[0]] + ' 十　万　零 ' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 十 ' + ReadPositiveInChinese(remainder)
    if nBit == 7:
        remainder = int(num) % 1000000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder)
        if remainder == 0:
            return numDict[numList[0]] + ' 百 万'
        if remainder < 10000:
            return numDict[numList[0]] + ' 百 万　零 ' + ReadPositiveInChinese(remainder)
        if remainder < 100000:
            return numDict[numList[0]] + ' 百 零 ' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 百 ' + ReadPositiveInChinese(remainder)
    if nBit == 8:
        remainder = int(num) % 10000000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder)
        if remainder == 0:
            return numDict[numList[0]] + ' 千 万'
        if remainder < 10000:
            return numDict[numList[0]] + ' 千 万 零 ' + ReadPositiveInChinese(remainder)
        if remainder < 1000000:
            return numDict[numList[0]] + ' 千 零　' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 千 ' + ReadPositiveInChinese(remainder)
    if nBit == 9:
        remainder = int(num) % 100000000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder)
        if remainder == 0:
            return numDict[numList[0]] + ' 亿'
        if remainder < 10000000:
            return numDict[numList[0]] + ' 亿 零 ' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 亿 ' + ReadPositiveInChinese(remainder)
    if nBit == 10:
        remainder = int(num) % 1000000000
        if numList[0] == '0':
            return numDict[numList[0]] + ' ' + ReadPositiveInChinese(remainder)
        if remainder == 0:
            return numDict[numList[0]] + ' 十 亿'
        if remainder < 100000000:
            return numDict[numList[0]] + ' 十 亿 零 ' + ReadPositiveInChinese(remainder)
        return numDict[numList[0]] + ' 十 亿 ' + ReadPositiveInChinese(remainder)
    # logger.info("Too big number {0} to read, maybe it is a mobile number".format(num))
    numList = list(str(num))
    s = str()
    for w in numList:
        s+= numDict[w] + ' '
    return s.strip()
def ReadPositiveInEnglish(num):
    if not (IsInteger(num) and  int(num) >= 0): return num
    verbatim = str()
    numDict = {0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
                   6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
                  11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
                  15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
                  19 : 'nineteen', 20 : 'twenty',
                  30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
                  70 : 'seventy', 80 : 'eighty', 90 : 'ninety'}
    if re.match(r'^0', str(num)):
        numList = list(str(num))
        for num in numList:
            verbatim += numDict[int(num)] + ' '
        return verbatim.strip()
    if num <= 20:
        return numDict[num]
    if num < 100:
        if num % 10 == 0:
            return numDict[num]
        else: 
            return numDict[num // 10 * 10] + '-' + numDict[num % 10]
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000
    if num < k:
        if num % 100 == 0: 
            return ( numDict[num // 100] + ' hundred')
        else: 
            return numDict[num // 100] + ' hundred and ' + ReadPositiveInEnglish(num % 100)
    if num < m:
        if num % k == 0: return ReadPositiveInEnglish(num // k) + ' thousand'
        else: return ReadPositiveInEnglish(num // k) + ' thousand ' + ReadPositiveInEnglish(num % k)
    if num < b:
        if (num % m) == 0: return ReadPositiveInEnglish(num // m) + ' million'
        else: return ReadPositiveInEnglish(num // m) + ' million ' + ReadPositiveInEnglish(num % m)
    
    for x in list(str(num)):
        verbatim += ReadPositiveInEnglish(int(x)) + ' '
    return verbatim.strip()
def ConvertEnglishNumber(s):
    numDict = {'zero':0, 'one':1,'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10,
               'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18,
               'nineteen':19, 'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'sixty':60, 'seventy':70, 'eighty':80,
               'ninety':90, 'hundred':100, 'thousand':1000, 'million':1000000, 'billion':1000000000}
    if s not in numDict: 
        m=re.search(r'^(\S+)\-(\S+)$', s)
        if m and m.group(1) in numDict and m.group(2) in numDict:
            return numDict[m.group(1)] + numDict[m.group(2)]
        return s
    return numDict[s]
def IsEnglishNumber(w):
    numDict = {'zero':0, 'one':1,'two':2, 'three':3, 'four':4, 'five':5, 'six':6, 'seven':7, 'eight':8, 'nine':9, 'ten':10,
               'eleven':11, 'twelve':12, 'thirteen':13, 'fourteen':14, 'fifteen':15, 'sixteen':16, 'seventeen':17, 'eighteen':18,
               'nineteen':19, 'twenty':20, 'thirty':30, 'forty':40, 'fifty':50, 'sixty':60, 'seventy':70, 'eighty':80,
               'ninety':90, 'hundred':100, 'thousand':1000,'million':1000000, 'billion':1000000000}
    if s not in numDict: 
        m=re.search(r'^(\S+)\-(\S+)$', s)
        if m and m.group(1) in numDict and m.group(2) in numDict:
            return True
        return False
    return True
def IsMagnitude(num):
    if num == 100 or num == 1000 or num == 1000000 or num == 1000000000:
        return True
    return False
def RollbackEnglishNumber(num):
    base1 = 100
    base2 = 1000
    if num == base1: return 'hundred'
    if num == base2: return 'thousand'
    if num == base2*base2: return 'million'
    if num == base2*base2*base2: return 'billion'
    return None
def CombineIntegerInEnglish(numList):
    if not numList or len(numList) == 1:
        return
    idx = 1
    length = len(numList)
    while idx < length:
        if IsInteger(numList[idx]) and IsInteger(numList[idx-1]):
            numList[idx-1] = str(numList[idx-1]) + str(numList[idx])
            del numList[idx]
            idx -= 1
            length -= 1
        idx += 1
def MergeNumberSegment(numList):
    length = len(numList)
    # sys.stderr.write("numList={0}\n".format(numList))
    idx = 1
    while idx < length:
        if IsInteger(numList[idx-1]) and IsInteger(numList[idx]):
            num1 = int(numList[idx-1])
            num2 = int(numList[idx])
            if IsMagnitude(num2):
                numList[idx-1] = num1*num2
                del numList[idx]
                idx -= 1
                length -= 1
            elif len(str(num1)) > len(str(num2)):
                numList[idx-1] = num1 + num2
                del numList [idx]
                idx -= 1
                length -= 1
        elif numList[idx-1] == 'point' and IsInteger(numList[idx]):
            jdx = idx
            w = '0.'
            while jdx < length and IsInteger(numList[idx]):
                w += str(numList[jdx])
                del numList[jdx]
                length -= 1
            numList[idx-1] = w
        elif numList[idx] == 'and':
            num1 = numList[idx-1]
            num2 = numList[idx+1]
            if len(str(num1)) > len(str(num2)):
                numList[idx-1] = num1 + num2
                del numList[idx]
                del numList[idx]
                idx -= 1
                length -= 2
        elif numList[idx] == 'point':
            jdx = idx + 1
            w = str()
            while jdx < length and IsInteger(numList[jdx]):
                w += str(numList[jdx])
                del numList[jdx]
                length -= 1
            numList[idx-1] = str(numList[idx-1]) + '.' + w
            del numList[idx]
            length -= 1
        idx += 1
def ConvertEnglishNumberInUtterance(utterance):
    wordList = re.split(r'\s+', utterance)
    transferList = list()
    newWordList = list()
    # sys.stderr.write("before={0}\n".format(utterance))
    for word in wordList:
        word1 = ConvertEnglishNumber(word)
        if word1 != word:
            if not transferList and RollbackEnglishNumber(int(word1)) :
                newWordList.append(str(word))
            else:
                transferList.append(word1)
        elif word1 == 'and' or word1 == 'point':
            if not transferList and word1 == 'and':
                newWordList.append(word1)
            else:
                if not transferList:
                    if word1 == 'point':
                        transferList.append(word1)
                    else:
                        newWordList.append(word1)
                else:
                    if IsInteger(transferList[-1]):
                        transferList.append(word1)
                    else:
                        newWordList.append(word1)
        else:
            trailingWord = str()
            if transferList:
                if transferList[-1] == 'and' or transferList[-1] == 'point':
                    trailingWord = transferList[-1]
                    del transferList[-1]
            if transferList:
                if len(transferList) == 1 and transferList[0] == 'point':
                    sys.stderr.write("utterance={0}\n".format(utterance))

                # retList = ReadSequenceNumber(transferList)
                MergeNumberSegment(transferList)
                CombineIntegerInEnglish(transferList)
                for word1 in transferList:
                    newWordList.append(str(word1))
            del transferList[:]
            if trailingWord: 
                newWordList.append(trailingWord)
            newWordList.append(word)
    if  transferList:
        trailingWord = str()
        # sys.stderr.write("transferList={0}\n".format(transferList))
        if transferList[-1] == 'and' or transferList[-1] == 'point':
            trailingWord = transferList[-1]
            del transferList[-1]
        if transferList:
            MergeNumberSegment(transferList)
            CombineIntegerInEnglish(transferList)
            for word1 in transferList:
                newWordList.append(str(word1))
        if trailingWord:
            newWordList.append(trailingWord)
    s = ' '.join(newWordList)
    # sys.stderr.write("after={0}\n".format(s))
    return s
            
def ReadDecimalWithBilingual(num, language):
    if not (IsInteger(num) and int(num) >= 0): return num
    numList = list(str(num))
    verbatim = str()
    for word in numList:
        if language.lower() == 'chinese':
            verbatim += ReadPositiveInChinese(int(word)) + ' '
        elif language.lower() == 'english':
            verbatim += ReadPositiveInEnglish(int(word)) + ' '
        else: raise Exception("Unknown language '{0}'".format(language))
    return verbatim.strip()

def ReadIntegerWithBilingual(word, language='chinese'):
    if not IsInteger(word): return word
    verbatim = str()
    if re.match(r'^0', word):
        if language.lower() == 'chinese':
            return ReadPositiveInChinese(word)
        if language.lower() == 'english':
            return ReadPositiveInEnglish(word)
    num = int(word)
    
    if num < 0: 
        num = -num
        if language.lower() == 'chinese':
            verbatim = '负 '
        elif language.lower() == 'english':
            verbatim = 'minus '
        else: raise Exception("Unknown language '{0}'".format(language))
    if language.lower() == 'chinese':
        verbatim += ReadPositiveInChinese(num)
    elif language.lower() == 'english':
        verbatim += ReadPositiveInEnglish(num)
    return verbatim
def ReadNormalNumberWithBilingual(word, language='chinese'):
    word0 = word
    word = str(word)
    # print("ReadNormalNumberWithBilingual:383: word={0}".format(word))
    verbatim = str()
    if re.match(r'^\-\d+', word): 
        if language.lower() == 'chinese':
            verbatim = '负 '
        elif language.lower() == 'english':
            verbatim = 'minus '
        word = re.sub(r'^\-', '', word)
    if re.match(r'^\+\d+', word):
        if language.lower() == 'chinese':
            verbatim = '正 '
        word = re.sub(r'^\+', '', word)
    word = re.sub(r',|\.$', '', word)
    if not IsRealNumber(word): return word0
    m = re.search(r'(\d+)?((\.\d+)?)', word)
    if m.group(1):
        verbatim += ReadIntegerWithBilingual(m.group(1), language)
    if m.group(2):
        s = m.group(2)
        s = re.sub(r'\.', '', s)
        if language.lower() == 'chinese':
            verbatim += ' 点 ' + ReadDecimalWithBilingual(s, language)
        elif language.lower() == 'english':
            verbatim += ' point ' + ReadDecimalWithBilingual(s, language)
    return verbatim
def ReadWithElipsis(verbatim):
    if re.search(r'百\s*(一|二|两|三|四|五|六|七|八|九)\s*十$', verbatim) or \
       re.search(r'千\s*(一|二|两|三|四|五|六|七|八|九)\s*百$', verbatim) or \
       re.search(r'万\s*(一|二|两|三|四|五|六|七|八|九)\s*千$', verbatim) or \
       re.search(r'亿\s*(一|二|两|三|四|五|六|七|八|九)\s*千\s*万$', verbatim):    
       return re.sub(r'(十|百|千|千\s*万)$', '', verbatim)
    return None
def ReadNormalNumberInChineseMutable(word):
    readDict = dict()
    verbatim = ReadNormalNumberWithBilingual(word, 'chinese')
    if verbatim != word:
        InsertDict(verbatim, readDict)
    '''
    if re.search(r'二\s*点|二\s*百|二\s*千|二\s*万|二\s*亿', verbatim):
        verbatim1 = re.sub(r'二', '两', verbatim)
        verbatim1 = re.sub(r'两\s*十', '二\s*十', verbatim1)
        verbatimList.append(verbatim1)
        verbatim2 = ReadWithElipsis(verbatim1)
        if verbatim2:
            verbatimList.append(verbatim2)
    verbatim1 = ReadWithElipsis(verbatim)
    if verbatim1:
        verbatimList.append(verbatim1, readDict)
    if re.search(r'^一\s*十', verbatim):
        verbatim1 = re.sub(r'^一\s*十', '十', verbatim)
        verbatimList.append(verbatim1)
    '''
    if IsInteger(word):
        verbatim = ReadDecimalWithBilingual(word, 'chinese')
        InsertDict(verbatim, readDict)
    '''
        if re.search(r'一', verbatim):
            verbatim = re.sub(r'一', '幺', verbatim)
            verbatimList.append(verbatim)
    '''
    readDict = MakeMutation(readDict,'chinese')
    return readDict
def IsYearNumber(word):
    """ recognize year 1600-2099 """
    m =re.match(r'^(1[6-9][0-9]{2}|20[0-9]{2})$', str(word))
    if not m: return False
    return True
def ReadYearNumber(s):
    if not IsYearNumber(s): return s
    charList = list(s)
    for idx, char in enumerate(charList):
        charList[idx] = ReadPositiveInChinese(char)
    return ' '.join(charList)
def InsertDict(word, targetDict):
    if word in targetDict:
        targetDict[word] += 1
    else: targetDict[word] = int(1)
def MakeMutation(targetDict, language='chinese'):
    tempDict = copy.deepcopy(targetDict)
    for word in targetDict:
        if language.lower() == 'chinese':
            if re.search(r'二\s*点|二\s*百|二\s*千|二\s*万|二\s*亿', word):
                word1 = re.sub(r'二', '两', word)
                word1 = re.sub(r'两\s*十', '二 十', word1)
                InsertDict(word1, tempDict)
                word2 = ReadWithElipsis(word1)
                if word2:
                    InsertDict(word2, tempDict)
            word3 = ReadWithElipsis(word)
            if word3:
                InsertDict(word3, tempDict)
            if re.search(r'^一\s*十', word):
                word4 = re.sub(r'^一\s*十', '十', word)
                InsertDict(word4, tempDict)
            if re.search(r'^(零\s*|一\s*|二\s*|三\s*|四\s*|五\s*|六\s*|七\s*|八\s*|九\s*)+$', word):
                word5 = re.sub(r'一', '幺', word)
                InsertDict(word5, tempDict)
        if language.lower() == 'english':
            if re.search('zero', word):
                word1 = re.sub('zero', 'o', word)
                # print("line301: word={0}, word1={1}".format(word, word1))
                InsertDict(word1, tempDict)
    return tempDict
def ReadNormalNumberInEnglishMutable(word):
    readDict = dict()
    verbatim = ReadNormalNumberWithBilingual(word, 'english')
    InsertDict(verbatim, readDict)
    verbatim = ReadDecimalWithBilingual(word, 'english')
    if verbatim != word:
        InsertDict(verbatim, readDict)
    if re.match(r'^[0-9]{2}[0-9]{2}$', word):
        m = re.search(r'(^[0-9]{2})([0-9]{2})$', word)
        last2 = m.group(2)
        prefix = ReadNormalNumberWithBilingual(m.group(1), 'english')
        if re.match('00', last2):
            verbatim = prefix + ' hundred'
            InsertDict(verbatim, readDict)
        elif re.match(r'0\d', last2):
            numList = list(str(last2))
            verbatim = prefix + ' o ' + ReadDecimalWithBilingual(numList[1], 'english')
            InsertDict(verbatim, readDict)
        else:
            verbatim = prefix + ' ' + ReadNormalNumberWithBilingual(m.group(2), 'english')
            InsertDict(verbatim, readDict)
    readDict = MakeMutation(readDict, "english")
    
    return readDict
def IsChineseNumber(word):
    if re.search(r'^(零|一|幺|壹|二|两|贰|三|叁|四|肆|五|伍|六|陆|七|柒|八|捌|九|玖|十|拾|百|佰|千|仟|万|萬|亿)+$', word):
        return True
    return False
def ReadSingleNumber(word):
    retNumber = -1
    cn2arabicDict = {u'零':0, u'一':1, u'幺':1, u'壹':1, u'二':2,u'两':2, u'贰':2, u'三':3, u'四':4, u'肆':4, u'五':5, u'伍':5, u'六':6, u'陆':6, u'七':7, u'柒':7, u'八':8, u'捌':8, u'九':9, u'玖':9}
    if word in cn2arabicDict:
        return cn2arabicDict[word]
    return retNumber
def Read10LevelNumber(word):
    retNumber = -1
    if not re.search(r'十|拾', word):
        return retNumber
    m = re.search(r'^(.*)(十|拾)(.*)$', word)
    if not m:
        return retNumber
    x = int(1)
    word1 = m.group(1).strip()
    if word1:
        x = ReadSingleNumber(word1)
        if x < 0: return retNumber
    x = x*10
    word3 = m.group(3).strip()
    if not word3: return x
    y = ReadSingleNumber(word3)
    if y < 0: return retNumber
    return x+y
    
def ReadHundredLevelNumber(word):
    retNumber = -1
    word0 = word
    if not re.search(r'百|佰', word):
        return retNumber
    m = re.search(r'^(.*)(百|佰)(.*)$', word)
    if not m:
        return retNumber
    word1 = m.group(1).strip()
    if not word1:
        return retNumber
    x = ReadSingleNumber(word1)
    if x < 0: return x
    x = x*100
    word3 = m.group(3).strip()
    if not word3:
        return x
    if re.search(r'^(零|〇)', word3):
        word3 = re.sub(r'^(零|〇)', '', word3)
        y = ReadSingleNumber(word3)
        if y < 0: return y
        return x + y
    nChar = len(word)
    if nChar == 1:
        word3 += '十'
    y = Read10LevelNumber(word3)
    if y < 0: return y
    return x + y
def ReadNumberLessThan1000(word):
    x = ReadHundredLevelNumber(word)
    if x > 0: return x
    x = Read10LevelNumber(word)
    if x > 0: return x
    x = ReadSingleNumber(word)
    return x        
def ReadThousandLevelNumber(word):
    retNumber = -1
    word0 = word
    if not re.search(r'千|仟', word):
        return retNumber
    m = re.search(r'^(.*)(仟|千)(.*)$', word)
    word1 = m.group(1).strip()
    word3 = m.group(3).strip()
    if not m: return retNumber
    if not word1: return retNumber
    x = ReadNumberLessThan1000(word1)
    if x < 0: return x
    x = x * 1000
    if not word3:
        return x
    if re.search(r'^(零|〇)', word3):
        word3 = re.sub(r'^(零|〇)', '', word3)
        y = ReadNumberLessThan1000(word3)
        if x < 0: return x
        return x+ y
    nChar = len(word3)
    if nChar == 1:
        word3 += '百'
    y = ReadNumberLessThan1000(word3)
    if y < 0: return y
    return x + y
def ReadNumberLessThan10000(word):
    x = ReadThousandLevelNumber(word)
    if x > 0: return x
    return ReadNumberLessThan1000(word)
def Read10ThousandLevelNumber(word):
    """ not efficient """
    retNumber = -1
    word0 = word
    if not re.search(r'万|萬', word):
        return retNumber
    m = re.search(r'^(.*)(万|萬)(.*)$', word)
    if not m:
        return retNumber
    word1 = m.group(1).strip()
    if not word1:
        return retNumber
    x = ReadNumberLessThan10000(word1)
    if x< 0: return retNumber
    x = x*10000
    word3 = m.group(3).strip()
    if not word3:
        return x
    if re.search(r'^零|^〇', word3):
        word3 = re.sub(r'^零|^〇', '', word3)
        y = ReadNumberLessThan1000(word3)
        if y < 0: return y
        return x + y
    if len(word3) == 1:
        word3 += '千'
    y = ReadNumberLessThan10000(word3)
    if y < 0:
        raise Exception("Unexpected number '{0}'".format(word0))
    return x + y
def ReadNumberLessThanYi(word):
    x = Read10ThousandLevelNumber(word)
    if x > 0: return x
    return ReadNumberLessThan10000(word)
def ReadUpToBillionNumber(word):
    retNumber = -1
    word0 = word
    if not re.search('亿', word):
        return retNumber
    m = re.search(r'^(.*)亿(.*)$', word)
    if not m:
        return retNumber
    word1 = m.group(1).strip()
    if not word1: return retNumber
    x  = ReadNumberLessThanYi(word1)
    if x < 0: return retNumber
    x = x*100000000
    word2 = m.group(2).strip()
    if not word2: return x
    if re.search(r'^零|^〇', word2):
        word2 = re.sub(r'^零|^〇', '', word2)
        y = ReadNumberLessThanYi(word2)
        if y < 0: return y
        return y + x
    if len(word2) == 1:
        word2 += '千万'
    y = ReadNumberLessThanYi(word2)
    if y < 0: return y
    return x + y
def ReadChineseNumber(word):
    if not re.search(r'(十|拾|百|佰|千|仟|万|萬|亿)', word):
        charList = list(word)
        numList = list()
        for nIndex, charWord in enumerate(charList):
            x = ReadSingleNumber(charWord)
            if x < 0: return word
            numList.append(str(x))
        word = ''.join(numList)
        return int(word)
    x = ReadNumberLessThan10000(word)
    if x > 0: return x
    x = Read10ThousandLevelNumber(word)
    if x > 0: return x
    x = ReadUpToBillionNumber(word)
    if x > 0: return x
    return word
        
def ReadChineseNumberList(wordList):
    word = ''.join(wordList)
    if not re.search(r'^零|〇|一|幺|壹|二|两|贰|三|叁|四|肆|五|伍|六|陆|七|柒|八|捌|九|玖|十|拾', word):
        return
    retWord = ReadChineseNumber(word)
    if retWord ==  word: return
    del wordList[:]
    wordList.append(str(retWord))
def PostProcessWordList(wordList):
    pass
def TransferChineseNumberInWordList(wordList, retWordList):
    tmpWordList = list()
    del retWordList[:]
    for nIndex, word in enumerate(wordList):
        if IsChineseNumber(word):
            if tmpWordList:
                nSize = len(tmpWordList)
                preIndex = tmpWordList[nSize-1][1]
                if preIndex + 1 != nIndex:
                    thisWordList = [item[0] for item in tmpWordList]
                    ReadChineseNumberList(thisWordList)
                    for thisWord in thisWordList:
                        retWordList.append(thisWord)
                    del tmpWordList[:]
            tmpWordList.append((word, nIndex))
        else:
            if tmpWordList:
                thisWordList = [item[0] for item in tmpWordList]
                ReadChineseNumberList(thisWordList)
                for thisWord in thisWordList:
                    retWordList.append(thisWord)
                del tmpWordList[:] 
            retWordList.append(word)
# the last one, if any
    if tmpWordList:
        thisWordList = [item[0] for item in tmpWordList]
        ReadChineseNumberList(thisWordList)
        for thisWord in thisWordList:
            retWordList.append(thisWord)
    PostProcessWordList(retWordList)
    return retWordList
def TransferChineseNumberInUtterance(utterance):
    wordList = utterance.split()
    transferWordList = list()
    TransferChineseNumberInWordList(wordList, transferWordList)
    utterance = ' '.join(transferWordList)
    return utterance
def IsDecreasedOrderSequence(numList):
    length = len(numList)
    idx = 0
    while idx < length -1:
        if IsInteger(numList[idx]) and IsInteger(numList[idx+1]):
            if len(numList[idx]) <= len(numList[idx+1]):
                return False
        else: return False
        idx += 1
    return True
    
def ConcatenateNumberInSequence(numList):
    length = len(numList)
    if length == 1:
        return str(numList[0])
    if IsDecreasedOrderSequence(numList):
        x = int(numList[0])
        idx = 1
        while  idx < length:
            x += int(numList[idx])
            idx += 1
        return str(x)
    return ''.join(numList)
def AppendNumberWord(wordList, cword):
    length = len(wordList)
    if length >=2:
        if wordList[-1] == 'and' and IsInteger(wordList[length-2]) and IsInteger(cword) and int(cword) < 100 and int(wordList[length-2]) >= 100:
            del wordList[-1]
            # sys.stderr.write("wordList[-1]={0}, cword={1}\n".format(wordList[-1], cword))
            wordList[-1] =str(int(wordList[-1])+ int(cword))
        elif wordList[-1] == 'point' and IsInteger(wordList[length-2]) and IsInteger(cword):
            del wordList[-1]
            wordList[-1] += '.' + str(cword)
        else: 
            wordList.append(cword)
    else:
        wordList.append(cword)
def PostProcessUtterance(utterance):
    wordList = re.split(r'\s+', utterance)
    numList = list()
    newWordList = list()
    for word in wordList:
        if IsInteger(word):
            numList.append(word)
        else:
            if numList:
                # sys.stderr.write("numList={0}\n".format(numList))
                word1 = ConcatenateNumberInSequence(numList)
                # sys.stderr.write("word1={0}\n".format(word1))
                # newWordList.append(word1)
                AppendNumberWord(newWordList, word1)
                del numList[:]
            newWordList.append(word)
    if numList:
        word1 = ConcatenateNumberInSequence(numList)
        AppendNumberWord(newWordList, word1)
        # newWordList.append(word1)
    utterance = ' '.join(newWordList)
    # utterance = re.sub(r'(\d+)\s+point\s+(\d+)', r'\1.\2', utterance)
    return utterance
        
def TransferEnglishNumberInUtterance(utterance):
    utterance = ConvertEnglishNumberInUtterance(utterance)
    return utterance
def ReadCurrencyNumberInChinese(word):
    if not IsCurrencyNumber(word): return word
    currency = '美元'
    if re.match(r'^￥', word): currency = '元'
    word0 = word
    word = re.sub(r'^\$', '', word)
    word = re.sub(r'^￥', '', word)
    word = re.sub(r',', '', word)
    word = NormalizeNumber(word)
    if not IsRealNumber(word): return word0
    utterance = ReadNormalNumberWithBilingual(str(word), 'chinese')
    if utterance == word: return word0
    return utterance + ' ' + currency

