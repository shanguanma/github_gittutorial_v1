#!/cm/shared/apps/python3.5.2/bin/python3.5
# -*- coding: utf-8 -*-

# Copyright 2017 mipitalk
#           2017 Zhiping Zeng
#           2017 Haihua Xu
from __future__ import print_function
import argparse
import sys
import os
import re
import codecs
import numpy as np
import logging
import random
from langconv import *
# sys.path.insert(0, 'steps')
# import libs.common as common_lib
sys.path.insert(0,'source-scripts/egs')
import python_libs.mipireadnumber as mipinumreader

logger = logging.getLogger('python_libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting crack-oov-word.py')

# begin utility
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
def _ReadPostiveInteger(num):
    convertDict = {0 : 'zero', 1 : 'one', 2 : 'two', 3 : 'three', 4 : 'four', 5 : 'five',
                   6 : 'six', 7 : 'seven', 8 : 'eight', 9 : 'nine', 10 : 'ten',
                  11 : 'eleven', 12 : 'twelve', 13 : 'thirteen', 14 : 'fourteen',
                  15 : 'fifteen', 16 : 'sixteen', 17 : 'seventeen', 18 : 'eighteen',
                  19 : 'nineteen', 20 : 'twenty',
                  30 : 'thirty', 40 : 'forty', 50 : 'fifty', 60 : 'sixty',
                  70 : 'seventy', 80 : 'eighty', 90 : 'ninety'}
    if num <= 20:
        return convertDict[num]
    if num < 100:
        if num % 10 == 0:
            return convertDict[num]
        else: 
            return convertDict[num // 10 * 10] + '-' + convertDict[num % 10]
    k = 1000
    m = k * 1000
    b = m * 1000
    t = b * 1000
    if num < k:
        if num % 100 == 0: 
            return ( convertDict[num // 100] + ' hundred')
        else: 
            return convertDict[num // 100] + ' hundred and ' + ReadIntegerWithEnglish(num % 100)
    if num < m:
        if num % k == 0: return ReadIntegerWithEnglish(num // k) + ' thousand'
        else: return ReadIntegerWithEnglish(num // k) + ' thousand ' + ReadIntegerWithEnglish(num % k)
    if num < b:
        if (num % m) == 0: return ReadIntegerWithEnglish(num // m) + ' million'
        else: return ReadIntegerWithEnglish(num // m) + ' million ' + ReadIntegerWithEnglish(num % m)
    return num
    # raise AssertionError("Number '{0}' is too large".format(str(num)))
def ReadIntegerWithEnglish(inStr):
    if not IsInteger(inStr):
        logger.info("unidentified integer string '{0}'".format(inStr))
        return inStr
    if int(inStr) < 0:
        logger.info("minus number '{0}'".format(inStr))
        return inStr
    return _ReadPostiveInteger(int(inStr))
    
    
def ReadDecimalWithEnglish(inStr):
    if not IsInteger(inStr):
        logger.info("identified integer string '{0}'".format(inStr))
        return inStr
    tmpList = list(inStr)
    retS = ''
    for sItem in tmpList:
        retS += ReadIntegerWithEnglish(sItem) + ' '
    retS = re.sub(r' $', '', retS)
    return retS

def IsCurrencyNumber(s):
    if not s: return False
    if not re.match(r'^\$\d+', s): return False
    return True
def ReadNormalNumber(word):
    utterance = str()
    word0 = word
    if re.match(r'^\-\d+', word): utterance = 'minus '
    word = re.sub(r'\-|,', '', word)
    word = re.sub(r'\.$|,$', '', word)
    if not IsRealNumber(word): return word0
    m = re.search(r'(\d+)?((\.\d+)?)', word)
    if m.group(1):
        utterance += str(ReadIntegerWithEnglish(m.group(1)))
    if m.group(2):
        numStr = m.group(2)
        numStr = re.sub(r'\.', '', numStr)
        utterance += ' point ' + ReadDecimalWithEnglish(numStr)
    return utterance
def ReadEnglishCurrencyNumber(word):
    if not IsCurrencyNumber(word):
        return word
    unit = 'dollars'
    word0 = word
    word = re.sub(r'[\$,]', '', word)
    utterance = ReadNormalNumber(word)
    if utterance == word: return word0
    utterance += ' dollars'
    return utterance
def IsPercentageNumber(word):
    if not word: return False
    if not re.search(r'%$', word): return False
    return True

def ReadEnglishPercentage(word):        
    if not IsPercentageNumber(word):
        return word
    if word == '%': return 'percent'
    word0 = word
    word = re.sub('%', '', word)
    utterance = ReadNormalNumber(word)
    if not utterance: return word0
    utterance += ' percent'
    return utterance
def ReadTimeNumber(word):
    m = re.match(r'^(\d+):(\d+)$', word)
    if not m: return word
    x = m.group(1)
    y = m.group(2)
    m = re.match(r'^0(\d)$', y)
    if m:
        y = m.group(1)
        return _ReadPostiveInteger(int(x)) + ' o ' + _ReadPostiveInteger(int(y))
    return _ReadPostiveInteger(int(x)) + ' ' + _ReadPostiveInteger(int(y))
def IsYearNumber(word):
    m =re.match(r'^(1[5-9][0-9]{2})$', str(word).strip())
    if m: return True
    else:
        m = re.match(r'^20[0-9]{2}$', str(word).strip())
        if m: return True
    return False
def ReadYearByFourNumber(word):
    if not IsYearNumber(word): return word
    numYear = int(word)
    if numYear >= 2000: return _ReadPostiveInteger(numYear)
    if numYear %100 ==0: return _ReadPostiveInteger(numYear)
    m = re.search(r'^(\d{2})(\d{2})$', str(numYear))
    x = m.group(1)
    y = m.group(2)
    m = re.search(r'^0(\d)$', y)
    if m:
        y = m.group(1)
        return _ReadPostiveInteger(int(x)) + ' oh ' + _ReadPostiveInteger(int(y))
    return _ReadPostiveInteger(int(x)) + ' ' + _ReadPostiveInteger(int(y))
def IsOrdinalNumber(word):
    if re.match(r"^\d+(st|nd|rd|th)", str(word)):
        return True
    return False
def _ReadOrdinal(n):
    convertDict = {0:'zeroth', 1:'first', 2:'second', 3:'third', 4:'fourth', 5:'fifth', 6:'sixth',
                   7:'seventh', 8:'eighth', 9:'ninth', 10:'tenth', 11:'eleventh', 12:'twelfth',
                   13:'thirteenth', 14:'fourteenth', 15:'fifteenth', 16:'sixteenth', 17:'seventeenth',
                   18:'eighteenth', 19:'nineteenth', 20:'twentieth', 30:'thirtieth', 40:'fortieth', 50:'fiftieth',
                   60:'sixtieth', 70:'seventieth', 80:'eightieth', 90:'ninetieth'}
    if n <= 20: return convertDict[n]
    if n < 100:
        if n % 10 == 0: return convertDict[n]
        else: return ReadIntegerWithEnglish((n//10)*10) + '-' + convertDict[n%10]
    k = 1000
    if n < k:
        if (n % 100) == 0: 
            return ReadIntegerWithEnglish(n//100) + ' hundredth'
        else:
            return ReadIntegerWithEnglish(n//100) + ' hundred and ' + _ReadOrdinal(n%100)
    m = k * 1000
    if n < m:
        if n % k == 0:
            return ReadIntegerWithEnglish(n//k) + ' thousandth'
        else:
            return ReadIntegerWithEnglish(n//k) + ' thousand ' + _ReadOrdinal(n%k)
    b = m * 1000
    if n < b:
        if n % m == 0:
            return ReadIntegerWithEnglish(n//m) + ' millionth'
        else:
            return ReadIntegerWithEnglish(n//m) + ' million ' + _ReadOrdinal(n % m)
    #    raise AssertionError("Number '{0}' is too large".format(word))
    return n
def ReadOrdinalNumber(word):
    if not IsOrdinalNumber(word): return word
    word0 = word
    word = re.sub(r'^(\d+)(st|nd|rd|th)$', r'\1', word)
    if not IsInteger(word): return  word0
    return _ReadOrdinal(int(word))
def ReadFractionNumber(word):
    m = re.search(r'^(\d+)\/(\d+)$', word)
    if not m: return word
    x = int(m.group(1))
    y = int(m.group(2))
    if x == 1 and y == 2: return 'half'
    if x == 24 and y == 7: return 'twenty-four seven'
    if x== 9 and y ==11: return 'nine eleven'
    if x == y : return _ReadPostiveInteger(x) + ' ' + _ReadPostiveInteger(x)
    if x == 1 and y >x: return _ReadPostiveInteger(x) + ' ' + _ReadOrdinal(y)
    if x > 1: return _ReadPostiveInteger(x) + ' ' + _ReadOrdinal(y) + 's'
    return word
def ReadRangeNumber(word):
    word0 = word
    word = re.sub(r'\+', '', word)
    m = re.search(r'(\d+)\-(\d+)$', word)
    if not m: return  word0
    tmpList = word.split("-")
    for x in tmpList:
        if not IsInteger(x): return word0
    nSize = len(tmpList)
    if nSize == 2:
       x0 = int(tmpList[0])
       x1 = int(tmpList[1])
       if IsYearNumber(x0):
           if IsYearNumber(x1):
               if x0 < x1:
                   return ReadYearByFourNumber(x0) + ' to ' + ReadYearByFourNumber(x1)
               elif x0%100 < x1: 
                   return ReadYearByFourNumber(x0) + ' to ' + _ReadPostiveInteger(x1)
       elif (x0 < x1 and x1 < 1000): 
            return _ReadPostiveInteger(x0) + ' to ' + _ReadPostiveInteger(x1)
       elif(x1 < 1000):
            return _ReadPostiveInteger(x0)  + ' ' + _ReadPostiveInteger(x1)
    word = str()
    for x in tmpList:
        word += ReadDecimalWithEnglish(str(x)) + ' '
    word = word.strip()
    return word            
def IsYearPlural(word):
    m = re.search(r'\d0s', word)
    if m: return True
    return False
def ReadYearPlural(word):
    if not IsYearPlural(word): return word
    word0 = word
    word =  re.sub(r"\'s", 's', word)
    m = re.search(r'^(\d+)s', word)
    if not m: return word0
    word = m.group(1)
    convertDict = {'ten':'teens', 'twenty':'twenties', 'thirty':'thirties', 'forty':'forties', 'fifty':'fifties',
                   'sixty':'sixties', 'seventy':'seventies', 'eighty':'eighties', 'ninety':'nineties', 'hundred':'hundreds', 'thousand':'thousands'}
    wordList = list()
    if re.match(r'^\d{4}$', word):
        if IsYearNumber(word):
            word = ReadYearByFourNumber(word)
        else:
            word = _ReadPostiveInteger(int(word))
    else:
        word = _ReadPostiveInteger(int(word))
    wordList = word.split() 
    nSize = len(wordList)
    lastWord = wordList[nSize-1]
    if not lastWord in convertDict:
        logger.info("word '{0}' is not in convertDict".format(lastWord))
        return word0
    wordList[nSize-1] = convertDict[lastWord]
    return ' '.join(wordList)
def _ReadYearPluralSimple(year):
    convertDict = {'ten':'teens', 'twenty':'twenties', 'thirty':'thirties', 'forty':'forties', 'fifty':'fifties',
                   'sixty':'sixties', 'seventy':'seventies', 'eighty':'eighties', 'ninety':'nineties', 'hundred':'hundreds', 'thousand':'thousands'}
    year = re.sub(r's$', '', year)
    if IsYearNumber(year):
        word = ReadYearByFourNumber(year)
    else:
        word = mipinumreader.ReadNormalNumberWithBilingual(year, 'english') 
    wordList = word.split()
    nSize = len(wordList)
    if not wordList[nSize-1] in convertDict:
        return year
    else:
        wordList[nSize-1] = convertDict[wordList[nSize-1]]
    return ' '.join(wordList)

def __ReadEnglishYearInterval(s):
    findList = re.findall(r'(\d+0s\s*(\-\s*\d+0s)?)', s)
    for pattern in findList:
        yearList = pattern[0].split('-')
        # sys.stderr.write("yearList={0}\n".format(yearList))
        word = _ReadYearPluralSimple(yearList[0])
        if word == yearList[0]:
            return s
        if len(yearList) == 1:
            s = s.replace(pattern[0], word)
        else:
            word += ' to'
            word1 = _ReadYearPluralSimple(yearList[1])
            if word1 == yearList[1]:
                return s
            else:
                word += ' ' + word1
            src = pattern[0]
            # sys.stderr.write("src={0}, dest={1}\n".format(src, word))
            s = s.replace(src, " {0} ".format(word))
    return s
def ReadCompositeWord(word, checkDict):
    if not re.match(r'.*\-.*', word): return word
    wordList = word.split("-")
    nwWord = 0
    ndWord = 0
    transWord = str()
    for xword in wordList:
        if re.match(r'^[a-z]+$', xword):
            nwWord += 1
            yword = xword
            if not yword in checkDict: yword = ' '.join(list(yword))
            transWord += yword + ' '
        elif re.match(r'^[\d]+$', xword): 
            ndWord += 1
            transWord += str(_ReadPostiveInteger(int(xword))) + ' '
    if nwWord == 0 or ndWord == 0: return word
    if nwWord + ndWord != len(wordList): return word
    return transWord.strip()
def ReadHybridWord(word, checkDict):
    if not (re.match(r'[a-z]+\d+$', word) or re.match(r'^\d+[a-z]+$', word)): return word
    if re.match(r'^\d+(th|st|rd|nd)$', word): return word
    if re.search(r'(0s)$', word): return word
    wordList = re.split(r'(\d+)', word)
    transfered = str()
    for xword in wordList:
        if not xword: continue
        if re.match(r'^\d+$', xword):
            transfered += _ReadPostiveInteger(int(xword)) + ' '
        else:
            yword = xword
            if yword not in checkDict: 
                yword = ' '.join(list(yword))
            transfered += yword + ' '
    return transfered.strip()
def ReadWordContainingPunct(word, checkDict):
    if not re.match(r'[a-z]+[\.,!;]([a-z]+)?', word): return word
    while re.search(r'([a-z]+)\.([a-z]+)', word):
        word = re.sub(r'([a-z]+)\.([a-z]+)', r'\1 dot \2', word)
    wordList = re.split(r'\.|,|!|;', word)
    transfered = str()
    for xword in wordList:
        if not xword: continue
        transfered += xword + ' '
    return transfered.strip()
def ReadAmpersandWord(word):
    if word == '&': return 'and'
    while re.search('([a-z]+|[0-9]+)&([a-z]+|[0-9]+)', word):
        word = re.sub('([a-z]+|[0-9]+)&([a-z]+|[0-9]+)', r'\1 and \2', word)
    return word
def IsOovWord(word, wordDict):
    if word not in wordDict:
        return True
    return False
# end utility
class OovNormalizer:
    def __init__(self, args):
        self._args = args
        self._wordDict = dict()
        self._oovDict = dict()
        self._textDict = dict()
        self._transferDict = dict()
        self._num_of_total_words = 0
        self._num_of_oov_words = 0
    def IsOovWord(self, word):
        if not self._wordDict:
            return False
        return IsOovWord(word, self._wordDict)
    def AddWordToOovDict(self, word):
        if not word in self._oovDict:
            self._oovDict[word] = int(1)
        else: self._oovDict[word] += 1
    def AddWordsToOovDict(self, words):
        num_of_oov = 0
        total_words = 0
        for word in words.split():
            word = word.strip()
            if not word: continue
            total_words += 1
            if self.IsOovWord(word):
                num_of_oov += 1
                self.AddWordToOovDict(word)
        self._num_of_total_words += total_words
        self._num_of_oov_words += num_of_oov
        
        return num_of_oov*100/total_words
    def DumpOovDict(self):
        retStr = str()
        for word, nCount in sorted(self._oovDict.items(), key = lambda x: int(x[1]), reverse=True):
            retStr += "{0:20} {1:4}".format(word, nCount) + "\n"
        return retStr
    def IsOovWordList(self, wordList):
        for word in wordList:
            if self.IsOovWord(word):
                return True
        return False
    def BuildLexicon(self, word, wordPron, tgtDict):
        if word in tgtDict:
            pronDict = tgtDict[word]
            if not wordPron in pronDict:
                pronDict[wordPron] = int(1)
        else:
            tgtDict[word] = dict()
            pronDict = tgtDict[word]
            pronDict[wordPron] = int(1)
    def GetWordDict(self): return self._wordDict
    def GetWordPron(self, word):
        if word not in self._wordDict: return None
        return list(self._wordDict[word].keys())
    def DumpWordDict(self):
        retstr = str()
        for word in self._wordDict:
            for pron in self.GetWordPron(word):
                retstr += "{0}\t{1}\n".format(word, pron)
        return retstr
    def DumpWordDict2(self, wordDict):
        s = str()
        for word in wordDict:
            for pron in wordDict[word]:
                s += "{0}\t{1}\n".format(word, pron)
        return s
    def LabelPronunciationForWordSequence(self, wordList, maximum=1):
        pronDict = dict()
        pronList = list()
        nWord = len(wordList)
        nIndex = 0
        backTrace = list()
        while(True):
            if nIndex < nWord:
                word = wordList[nIndex]
                if self.IsOovWord(word):
                    logger.info("oov word '{0}' seen".format(word))
                    return pronDict
                curPronList = self.GetWordPron(word)
                curPron = curPronList[0]
                # we only get one
                if maximum == 1:
                    del curPronList[:]
                    curPronList.append(curPron)
                backTrace.append([curPronList, 1])
                pronList.append(curPron)
                nIndex += 1
            elif nIndex == nWord:
                tmpPron = ' '.join(pronList)
                tmpWord = ' '.join(wordList)
                self.BuildLexicon(tmpWord, tmpPron, pronDict)
                nIndex -= 1
                del pronList[-1]
                [curPronList, nPron] = backTrace.pop()
                nTotalPron = len(curPronList)
                while(nIndex >=0 and nPron >= nTotalPron):
                    nIndex -= 1
                    if nIndex < 0:
                        return pronDict
                    pronList.pop()
                    [curPronList, nPron] = backTrace.pop()
                    nTotalPron = len(curPronList)
                curPron = curPronList[nPron]
                pronList.append(curPron)
                backTrace.append([curPronList, nPron+1])
                nIndex += 1
    def WriteArabicNumberLexicon(self, word, arabicDict):
        word = str(word)
        if not re.match(r'(\d+(\.\d+)?)', word): return None
        cnVerbatimDict = mipinumreader.ReadNormalNumberInChineseMutable(word)
        enVerbatimDict = mipinumreader.ReadNormalNumberInEnglishMutable(word)
        retDict = cnVerbatimDict.copy()
        retDict.update(enVerbatimDict)
        for wordsequence in retDict:
            wordList = re.split(r'[\s\-]', wordsequence)
            wordList = list(filter(None, wordList))
            # logger.info('wordList={0}'.format(wordList))
            labelDict = self.LabelPronunciationForWordSequence(wordList)
            if not labelDict:
                logger.info("word={0}, cnVerbatimDict={1}".format(word, cnVerbatimDict))
                continue
            for ws in labelDict:
                newWord = "{0} [{1}]".format(word, ws)
                for pron in labelDict[ws]:
                    self.BuildLexicon(newWord, pron, arabicDict)
    def LoadDicts(self):
        if self._args.oov_word_count_file:
            with open(self._args.oov_word_count_file) as istream:
                for line in istream:
                    wList = [x.strip() for x in line.split()]
                    if len(wList) == 2:
                        self._oovDict[wList[0]] = wList[1]
                    else:
                        logger.warn("bad line {0}".format(line))
        if self._args.word_lexicon_file:
            with open(self._args.word_lexicon_file, encoding='utf-8') as istream:
                for line in istream:
                    wList = [x.strip() for x in line.split()]
                    word = wList.pop(0)
                    wordPron = ' '.join(wList).strip()
                    self.BuildLexicon(word, wordPron, self._wordDict)
            logger.info("done with loading lexicon file: '{0}'".format(self._args.word_lexicon_file))
        # if self._args.text_file:
        #    with open(self._args.text_file, encoding='utf-8') as istream:
        #        for line in istream:
        #            line = re.sub(r'\s+', '', line.strip())
        #            if line and not line in self._textDict:
        #                self._textDict[line] = int(1)
        #            else:
        #                self._textDict[line] += 1
        if self._args.word_transfer_dict_file:
            with open(self._args.word_transfer_dict_file, encoding='utf-8') as istream:
                for line in istream:
                    m = re.search(r'^(\S+)\s+(.*)$', line)
                    if not m: continue
                    word = m.group(1)
                    transfer = m.group(2)
                    if word in self._transferDict:
                        logger.info("Duplicated line '{0}' in transfer dict".format(line))
                        continue
                    self._transferDict[word] = transfer
    def TransferWords(self, words):
        wordList = words.split()
        if not wordList: return None
        newWords = str()
        for word in wordList:
            word = ReadYearPlural(word)
            # word = ReadYearByFourNumber(word)
            if word in self._transferDict:
                word = self._transferDict[word]
            newWords += word + ' '
        newWords = re.sub(r'\s+', ' ', newWords)
        return newWords.strip()
    def GetLabelAndWords(self, line, lowercase=False):
        m = re.search(r'^(\S+)\s+(.*)$', line)
        retList = list()
        if not m: return None
        label = m.group(1)
        words = m.group(2).strip()
        if not words: return None
        retList.append(label)
        if lowercase: words = words.lower()
        retList.append(words)
        return retList
    def ReadOovWordList(self):
        if self._args.oov_word_count_file:
            for word, nCount in sorted(self._oovDict.items(), key = lambda x: int(x[1]), reverse=True):
                transfered = ReadEnglishCurrencyNumber(word)
                if word == transfered:
                    transfered = ReadAmpersandWord(word)
                if word == transfered:
                    transfered = ReadEnglishPercentage(word)
                if word == transfered:
                    transfered = ReadOrdinalNumber(word)
                if word == transfered:
                    transfered = ReadFractionNumber(word)
                if word == transfered:
                    transfered = ReadTimeNumber(word)
                if word == transfered:
                    transfered = ReadYearByFourNumber(word)
                if word == transfered:
                    transfered = ReadRangeNumber(word)
                if word == transfered:
                    transfered = ReadYearPlural(word)
                if word == transfered:
                    transfered = ReadCompositeWord(word, self._wordDict)
                if word == transfered:
                    transfered = ReadHybridWord(word, self._wordDict)
                if word == transfered:
                    transfered = ReadWordContainingPunct(word, self._wordDict)
                if word == transfered:
                    transfered = ReadNormalNumber(word)
                print("{0:20}\t{1:4}\t{2}".format(word, nCount, transfered))
    def CollectOovOfTransferDict(self):
        oovWordDict = dict()
        if not self._wordDict:
            raise Exception("word dict is not loaded, try to use --word-lexicon-file to load word dict")
        if self._args.word_transfer_dict:
            with open(self._args.word_transfer_dict, encoding='utf-8') as istream:
                for line in istream:
                    m = re.search(r'^(\S+)\s+(\d+)\s+(.*)$', line)
                    if not m: continue
                    nCount = m.group(2)
                    transfered = m.group(3)
                    if not transfered: continue
                    wordList = re.split(r'[\s\-]', transfered)
                    for word in wordList:
                        if not word in self._wordDict:
                            if not word in oovWordDict:
                                oovWordDict[word] = int(nCount)
                            else:
                                oovWordDict[word] += int(nCount)
            for word, nCount in sorted(oovWordDict.items(), key = lambda x:x[1], reverse=True):
                print("{0:4}\t{1}".format(nCount, word))
        

def get_args():
    parser = argparse.ArgumentParser(description='Arguments_parser')
    parser.add_argument('--oov-word-count-file', dest='oov_word_count_file', type=str, help='file record oov count info')
    parser.add_argument('--word-lexicon-file', dest='word_lexicon_file', type=str, help='lexicon file')
    parser.add_argument('--text-file', dest='text_file', type=str, help='text_file that oov words come from')
    parser.add_argument('--target-dir', dest='target_dir', type=str, help='target exp dir')
    parser.add_argument('--transfer-dict-file', dest='word_transfer_dict_file', type=str, help='word transfer dict')
    parser.add_argument('--read-oov-for-transfer-dict', dest='read_oov_for_transfer_dict', action='store_true')
    parser.add_argument('--collect-oov-from-transfer-dict', dest='collect_oov_from_transfer_dict', action='store_true')
    parser.add_argument('--tgtdir', dest='tgtdir', type=str, help='target folder for output if any')
    parser.add_argument('--normalize-mipitalk-mobile-data', dest='normalize_mipitalk_mobile_data', action='store_true')
    parser.add_argument('--normalize-text', dest='normalize_text', action='store_true')
    parser.add_argument('--normalize-text-with-uttid', dest='normalize_text_with_uttid', action='store_true')
    parser.add_argument('--transfer-text', dest='transfer_text', action='store_true')
    parser.add_argument('--dump-oov-of-text', dest='dump_oov_of_text', action='store_true')
    parser.add_argument('--label-pronunciation-for-number-word', dest='label_pronunciation_for_number_word', action='store_true')
    parser.add_argument('--oov-thresh-in-utterance', dest='oov_thresh_in_utterance', type=int, default=3, help='maximum oov words permitted in each utterances')

    parser.add_argument('--test', dest='test', action='store_true')
    parser.add_argument('--language', dest='language', type=str.lower, default='unknown')
    parser.add_argument('--test-chinese', dest='test_chinese', action='store_true')
    parser.add_argument('--test-english', dest='test_english', action='store_true')
    parser.add_argument('--test-chinese-segmentation', dest='test_chinese_segmentation', action='store_true')
    parser.add_argument('--test-label-pronunciation-for-wrod-sequence', dest='test_label_pronunciation_for_word_sequence', action='store_true', help='test word sequence pronunciation labeling')
    parser.add_argument('--test-read-arabic-number-in-english', dest='test_read_arabic_number_in_english', action='store_true')
    parser.add_argument('--write-arabic-number-lexicon', dest='write_arabic_number_lexicon', action='store_true')
    return parser.parse_args()
def TestReadArabicNumberInEnglish(args):
    translateDict = dict()
    number1 = '$21'
    print("number={0}, translated={1}".format(number1, ReadEnglishCurrencyNumber(number1)))
    number2 = '51.4%'
    print("number={0}, translated={1}".format(number2, ReadEnglishPercentage(number2)))
    for x in [1, 2, 3, 4, 5, 10, 100, 102, 1002, 100002, 123]:
        print("number={0}, ordinal={1}".format(x, _ReadOrdinal(x)))
    for x in ["1st", "2nd","3rd","4th","5th","10th","100th", "102nd", "1002nd", "100002nd"]:
        print("number={0}, ordinal={1}".format(x, ReadOrdinalNumber(x)))
    for x in ['1/2', '2/3', '9/11', '20/20', '24/7' ]:
        print("number={0}, fration={1}".format(x, ReadFractionNumber(x)))
    for x in ['1:2', '2:05', '7:09','10:40']:
        print ("number={0}, transfer={1}".format(x, ReadTimeNumber(x)))
        for x in [1605, 1905, 1657,2011,2016,2015,2017,2014,2013,2000,2010,2012,2008,1965,2009,2001,2003,1000,2007,1991,2004,2006,2020,4000,1500]:
            print ("number={0}, transfer={1}".format(x, ReadYearByFourNumber(x)))
    for x in [ '10-15', '669-1193', '20-30', '669-1938', '691-1938', '3-4', '+1-800-255-0000', '2-3', '5-10', '15-20', '1-1', '50-50', '2016-2017', '30-40', '3-1', '9-11', '9-3', '10-20', '10-4', '20-25', '40-50', '4-5', '669-119-3821', '7-8' ]:
        print ("word={0}, transfer={1}".format(x, ReadRangeNumber(x)))
        for x in ["80s","90s","1980s","60s","70s","1960s","1970s","50s","1990s","1950s","20s","30s","2000s","40s","1920s","1930s","1940s","1800s","1880s","1900s","100s","1600s","1840s","10s","1700s","1780s","1790s","1830s","1890s","200s","2980s","360s","570s"]:
            print ("number={0}, transfer={1}".format(x, ReadYearPlural(x)))
        for x in [ "f-35","i-10","ar-15","i-20","a-1","c-130","omega-3","one-77","ak-47","br-2","c-123","carbon-12","catch-22","f-150","i-17","i-25","i-27","i-35","i-65","i-70","i-81","in-37","mx-5","perrin-410","sg-1","su-50","a-10","ar-22","b-17","b-24","b-52","beta-2","blink-182","c-17","ch-47","csf-3","diane-35","dm-810","dsm-5","e-40","eco-1","ecs-12","end-2016","euro-6","f-16","ffx-2","fm-544","gc-2","gs-1","gs-7","h-60","hi-5","hu-100","i-15","i-380","i-39","i-394","i-40","i-5","i-55","i-57","i-74","i-76","i-77","i-85","i-90","icd-10","jp-4","july-2014","k-30","l-188","m-43","mid-2015","mp-35","ms-13","mxvi-4000","ny-32","p-8","pac-12","par-5","pcg-10","pon-3","pro-1","pt-2030","spp-1","sr-71","sub-2","t-1000","tb-1","tier-2","tv-14","type-1","type-2","u-22","u-32","uc-11","under-12","under-19","under-23","uo-1","us-224","vp-1","vp-30","vr-57","vsx-52","x-360","xv-700","24-hour","7-eleven","10-year","10-minute","20-something","36-hour","30-year","45-minute","20-year","360-degree","10-day","10-second","25-year","30-day"]:
            print ("number={0}, transfer={1}".format(x, ReadCompositeWord(x, dict())))
        for x in ["3d", "4k", "938live", "2g", "4g", "10k", "3g", "5c", "2d", "2k", "5s", "ps4", "f1", "sg100", "co2", "ps3", "s5", "fy17", "mp4"]:
            print("number={0}, transfered={1}".format(x, ReadHybridWord(x,dict())))
        for x in ["dr.", "p.m.", "u.s.", "a.m.", "that.", "facebook.com", "it.", "this.", "you.", "me.", "vs.", "okay.", "now.", "well.", "it,", "the.", "too.", "red.", "google,", "time.", "on.", "one.", "up.", "us.", "again.", "long.", "so,", "so.", "then.", "right.", "that,", "a,", "go.", "like.", "nice.", "okay,", "today.", "morning.", "much.", "be.", "know.", "thing.", "way.", "all.", "campaign.", "mediacom.com"]:
            print ("number={0}, transfered={1}".format(x, ReadWordContainingPunct(x,dict())))

def NormalizeUtteranceContainChineseWords(words, normalizer):
    words = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fa5])", r'\1 \2', words)
    words = re.sub(r"([\u4e00-\u9fa5])([A-Z]a-z0-9)", r'\1 \2', words)
    wordList = words.split()
    words = str()
    for word in wordList:
        if normalizer.IsOovWord(word) and re.search(r'[\u4e00-\u9fa5]', word):
            charList = list(word)
            for char in charList:
                if normalizer.IsOovWord(char):
                    normalizer.AddWordToOovDict(char)
            words += ' '.join(charList) + ' '
        elif normalizer.IsOovWord(word):
            normalizer.AddWordToOovDict(word)
            words += word + ' '
        else: words += word + ' '
    return words.strip()
def ReadAmPmTimeLikeWord(word):
    oldWord = word
    if re.match(r'(\d+(\.\d+)?)([a-z]+)$', word):
        match = re.search(r'(\d+\.\d+)([a-z]+)$', word)
        string = str()
        word = str()
        if match:
            numStr = match.group(1)
            string = match.group(2)
            strList = re.split(r'\.', numStr)
            word += ReadIntegerWithEnglish(strList[0])
            if re.match(r'^(am|pm)$', string):
                if int(strList[1]) != 0:
                    word += ' ' + ReadIntegerWithEnglish(strList[1])
            else:
                word += ' point ' + ReadDecimalWithEnglish(strList[1])
        else:
            # logger.info("word={}".format(oldWord))
            match= re.search(r'(\d+)([a-z]+)$', oldWord)
            word += str(ReadIntegerWithEnglish(match.group(1)))
            string = match.group(2)
        word += ' ' + string
        # logger.info("old word= '{0}', word={1}".format(oldWord, word))
    return word
def NormalizeEnglishUtterance(words, normalizer):
    wordList = words.split()
    words = str()
    for word in wordList:
        word1 = ReadEnglishCurrencyNumber(word)
        if word1 == word:
            word1 = ReadOrdinalNumber(word)
        if word1 == word:
            word1 = ReadYearPlural(word)
        if word1 == word:
            word1 = ReadCompositeWord(word, normalizer.GetWordDict())
        if word1 == word:
            word1 = ReadAmPmTimeLikeWord(word)
        words += word1 + ' '
    wordList = words.strip().split()
    for word in wordList:
        if normalizer.IsOovWord(word):
            normalizer.AddWordToOovDict(word)
    words = ' '.join(wordList)
    return words.strip()
def NormalizeMipitalkMobileData(args):
    oovNormalizer =OovNormalizer(args)
    oovNormalizer.LoadDicts()
    outputLines = str()
    if args.text_file:
        with open(args.text_file, encoding='utf-8') as istream:
            for line in istream:
                m = re.search(r'^(\S+)\s+(.*)$', line)
                if not m: continue
                label = m.group(1)
                words = m.group(2).strip()
                if not words: continue
                if re.search(r'[\u4e00-\u9fa5]', line):
                    words = NormalizeUtteranceContainChineseWords(words, oovNormalizer)
                else:
                    words = NormalizeEnglishUtterance(words, oovNormalizer)
                outputLines += label + ' ' + words + "\n"
    if args.tgtdir:
        # if not os.path.exists(args.tgtdir):
        #    os.makedirs(args.tgtdir)
        ostream = open("{0}".format(args.tgtdir), 'w', encoding='utf-8')
        ostream.write(outputLines)
        ostream.close()
'''
        oovStrs = oovNormalizer.DumpOovDict()
        ostream = open("{0}/oov-word-count.txt".format(args.tgtdir), 'w', encoding='utf-8')
        ostream.write(oovStrs)
        ostream.close()        
'''
def NormalizeWhiteSpace(s):
    s = re.sub(r"^\s+|\s+$", '', s)
    s = re.sub(r"\s+", ' ', s)
    return s
def strQ2B(ustring):
    """把字符串全角转半角 obtained with baidu"""
    rstring = ""
    for uchar in ustring:
        inside_code=ord(uchar)
        if inside_code==0x3000:
            inside_code= 32
        elif inside_code >= 65281 and inside_code <= 65374:   #转完之后不是半角字符返回原来的字符
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
def ConvertT2SChinese(s): 
    retList = list()
    for w in [w for w in s.split() if w ]:
        w1 = strQ2B(w)
        w1 = Converter('zh-hans').convert(w1)
        retList.append(w1)
    return ' '.join(retList)
def ReadYear(s):
    m = re.search(r'(^\d{4}$)', s)
    if not m: return s
    digitList = list(m.group(1))
    for idx, digit in enumerate(digitList):
        digitList[idx] = mipinumreader.ReadPositiveInChinese(digit)
        year = ' '.join(digitList)
    return year
        
def ConvertYear(s):
    m = re.search(r'(\d{4})\s+(年)', s)
    while m:
        sstr = m.group(1)
        year = ReadYear(sstr)
        s = s.replace(m.group(1), year)
        m = re.search(r'(\d{4})\s+(年)', s)
    return s
def IsChineseWord(w):
    m = re.match(r'^[\u4e00-\u9fa5]+$', w)
    if m: return True
    return False
def IsEnglishWord(w):
    # if w.isalpha(): return True
    if re.match(r'^[a-z]+$', w): return True
    if re.match(r'^[a-z]+[0-9]+$', w): return True
    if re.match(r'^[a-z]+\'s$', w): return True
    if re.match(r'^[a-z]+s\'$', w): return True
    if re.match(r'^[a-z]+\'[a-z]+$', w): return True
    return False
def ReadDateString(w):
    m = re.search(r'^(\d{4})\-(\d{2})\-(\d{2})$', w)
    w1 = str()
    if not m: return w
    w1 = ReadYear(m.group(1)) + ' 年 '
    w1 += mipinumreader.ReadPositiveInChinese(m.group(2)) + ' 月 '
    w1 += mipinumreader.ReadPositiveInChinese(m.group(3)) + ' 日'
    return w1
def SegmentChineseUtterance (s):
    s = re.sub(r"([\u4e00-\u9fa5])([\u4e00-\u9fa5])", r' \1 \2 ', s)
    s = re.sub(r"([\u4e00-\u9fa5])([A-Za-z0-9])", r'\1 \2', s)
    s = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fa5])", r'\1 \2', s)
    s = re.sub(r'([\u4e00-\u9fa5])\s*(·|‘|’|:|：)', r' \1 ', s)
    return s
def ReadContextIndependentNumber(snum):
    m = re.match(r'^\d+$', str(snum))
    if not m: return snum
    nDigit = len(str(snum))
    if nDigit < 6:
        return snum
    digitList = list(str(snum))
    for idx, digit in enumerate(digitList):
        digitList[idx] = mipinumreader.ReadPositiveInChinese(digit)
    snum = ' '.join(digitList)
    return snum
def IsTimeString(l):
    N = len(l)
    if(N<2): return False
    if l[0] >= 24 or l[1] >= 60 or int(l[0]) < 10 and int(l[1]) < 10: return False
    if N > 2:
        if l[2] >= 60: return False
    return True
def ReadTimeNumberInChinese(s):
    ''' to be rewritten '''
    s1 = str()
    for w in s.split():
        m = re.match(r'^(\d+):(\d+):(\d+)$', w)
        if m:
            l = list()
            l.append( int(m.group(1)))
            l.append(int(m.group(2)))
            l.append( int(m.group(3)))
            w1 = str()
            if IsTimeString(l):
                w1 = mipinumreader.ReadPositiveInChinese(l[0]) + ' 点' + ' '
                w1 += mipinumreader.ReadPositiveInChinese(l[1]) + ' 分' + ' '
                w1 += mipinumreader.ReadPositiveInChinese(l[2]) + ' 秒'
                s1 += w1 + ' '
            else:
                w1 = mipinumreader.ReadPositiveInChinese(l[0]) + ' 比' + ' '
                w1 += mipinumreader.ReadPositiveInChinese(l[1]) + ' 比' + ' '
                w1 += mipinumreader.ReadPositiveInChinese(l[2])
                s1 += w1 + ' '
            continue
        m = re.match(r'^(\d+):(\d+)$', w)
        if m:
            l = list()
            l.append( int(m.group(1)))
            l.append(int(m.group(2)))
            w1 = str()
            if IsTimeString(l):
                w1 = mipinumreader.ReadPositiveInChinese(l[0]) + ' 点' + ' '
                w1 += mipinumreader.ReadPositiveInChinese(l[1]) + ' 分'
                s1 += w1 + ' '
            else:
                w1 = mipinumreader.ReadPositiveInChinese(l[0]) + ' 比' + ' '
                w1 += mipinumreader.ReadPositiveInChinese(l[1])
                s1 += w1 + ' '
            continue
        s1 += ReadDateString(w) + ' '
    return s1.strip()
def ConvertArabicToChineseNumber(s):
    s1 = str()
    sys.stderr.write("ConvertArabicToChineseNumber(874): s={0}\n".format(s))
    for w1 in s.split():
        if IsInteger(w1):
            w2 = mipinumreader.ReadYearNumber(w1)
            if w1  == w2:
                w2 =mipinumreader.ReadNormalNumberWithBilingual(w1, 'chinese')
            s1 += ' ' + w2
        elif IsRealNumber(w1):
            s1 += ' ' + mipinumreader.ReadNormalNumberWithBilingual(w1, 'chinese')
        else:
            s1 += ' ' + w1
    # sys.stderr.write("ConvertArabicToChineseNumber(886):s1={0}\n".format(s1))
    return s1.strip()
def ConvertEnglishNumberToArabicNumber(s):
    return mipinumreader.TransferEnglishNumberInUtterance(s)
def NormalizePercentData(s):
    s = re.sub(r'(\d+)\s+(\d+)\s*%', r' \1.\2% ', s)
    s = re.sub(r'(\d+\.\d+)\s+%', r'\1%', s)
    s = re.sub(r'(\d+)\s+%', r'\1%',s)
    s = re.sub(r'(\d+\.\d+)\s+percents', r'\1%', s)
    s = re.sub(r'(\d+)\s+percents', r'\1%', s)
    s = re.sub(r'(\d+)\s+(\d+)\s+per cent', r'\1.\2%', s)
    s = re.sub(r'(\d+)\s+per cent', r'\1%', s)
    m = re.search(r'(\S+)%', s)
    while m:
        w1 = m.group(1)
        if IsRealNumber(w1):
            w2 = '百分之' + ' '
            w2 += mipinumreader.ReadNormalNumberWithBilingual(w1, 'chinese')
            s = s.replace('{0}%'.format(w1), w2)
        else:
            s = s.replace('{0}%'.format(w1), '百分之 {0} '.format(w1))
        m = re.search(r'(\S+)%', s)
    return s
def RemovePunctuation(s):
    s = re.sub(r';|"', '\n', s)
    s = re.sub(r'「|」',' ', s)
    s = re.sub(r'([\u4e00-\u9fa5]+)\s*(《|》|、|“|”|\(|\)|\"|\'|:|;|›)', r' \1 ', s)
    s = re.sub(r'(《|》|、|“|”|\(|\)|\"|\'|:|;|›)\s*([\u4e00-\u9fa5]+)', r' \2 ', s)
    s = re.sub(r'([\u4e00-\u9fa5]+)\s*(《|》|、|”|\(|\)|\"|\'|:|;|›|”|“)', r' \1 ', s)
    s = re.sub(r'([\u4e00-\u9fa5]+)\s*(。|,|!|\?|❓)', r' \1 <s> ', s)
    s = re.sub(r'([a-z]+)\s*(、|。|,|”|“|》)', r'\1 ', s)
    s = re.sub(r'\.{3,6}', ' ', s)
    s = s.replace('……', ' ')
    # if re.search('%', s):
    #    logger.info("utterance={0}".format(s))
    s = re.sub(r'(\d+)\.([^\d]+)', r'\1 \2', s)
    
    s = re.sub(r'(\d+)\.$', r'\1', s)
    return s
def NormalizePercentNumber(s):
    s = re.sub(r'(\d+)\s+(\d+)\s+percent', r'\1.\2%', s)
    s = re.sub(r'(\d+)\s+(\d+)\s+ percents', r'\1.\2%', s)
    s = re.sub(r'(\d+)\s+percent', r'\1%', s)
    s = re.sub(r'(\d+)\s+per cent', r'\1%', s)
    s = re.sub(r'(\d+)\s+(\d+)\s*%', r'\1.\2%', s)
    s = re.sub(r'(\d+)\s+%', r'\1%', s)
    return s
def NormalizeCurrencyNumber(s):
    s = re.sub(r'\$\s*(\d+)\s+(\d+)', r'$\1\2', s)
    s = re.sub(r'\$\s+(\d+)', r'$\1', s)
    s = re.sub(r'£\s*(\d+)\s+(\d+)', r'£\1\2', s)
    s = re.sub(r'£\s+(\d+)', r'£\1', s)
    s = re.sub(r'(\d+)\s+(\d+)bn', r'\1.\2 billion', s)
    s = re.sub(r'(\d+)bn', r'\1 billion', s)
    s = re.sub(r'(\d+)\s+(\d+)m', r'\1.\2 million', s)
    s = re.sub(r'(\d+)m', r'\1 million', s)
    return s
def ConvertChineseIntegerClass(s):
    m = re.search(r'\s+(\d+)\s+(月|日|号|个|人|岁|件|口|天|多 年|次|亿|多 人|万|枚|元)', s)
    while m:
        number =  mipinumreader.ReadPositiveInChinese(m.group(1))
        if number == m.group(1):
            return s
        s = re.sub(r'\s+(\d+)\s+(月|日|号|个|人|岁|件|口|天|多 年|次|亿|多 人|万|枚|元)', r' {0} \2'.format(number), s)
        m = re.search(r'\s+(\d+)\s+(月|日|号|个|人|岁|件|口|天|多 年|次|亿|多 人|万|枚|元)', s)
    return s
def ConvertChineseRealClass(s):
    """ The following two lines are particularly for abnormal text """
    return s
    # s = re.sub(r'(\d+)\s+(\d+)\s+(亿|万|平 方|毫 米|米|克|毫 克|公 里|公 尺|公 斤|公 顷|元|美 元|英 镑|秒|小 时|分|度)', r'\1.\2 \3', s)
    # s = re.sub(r'(北 纬|东\s*经|提 高|增 长|降 至|下 跌|下 挫|升 至|占 比|占 比 为)\s+(\d+) (\d+)', r'\1 \2.\3', s)
    # number = mipinumreader.ReadNormalNumberWithBilingual(str(12.3), 'chinese')
    # return s
def NormalizeNormalNumber(s):
    while re.search(r'\d+,\d{3}', s):
        s = re.sub(r'(\d+),(\d{3})', r'\1\2', s)
    # while re.search(r'(\d+)(,|:|;|!|")', s):
    #    s = re.sub(r'(\d+)(,|:|;|!|")', r'\1 \2', s)
    return s
def ConvertChineseData(s):
    s = ConvertYear(s)
    s = ConvertChineseIntegerClass(s)
    s = ConvertChineseRealClass(s)
    return s
def ReadTemperature(s, language):
    s = s.replace('℃ ', '°C')
    m = re.findall(r'(((\-)?\d+(\.\d+)?\-)?(\-)?(\d+(\.\d+)?))(°C|°F)', s)
    for e in m:
        pattern = e[0] + e[7]
        # logger.info("e={0}, length_of_e={1}".format(pattern, len(e)))
        sList = s.split(pattern)
        contextLanguage = GetLanguageContextByStrPair(sList, language)
        word = str()
        if e[1]:
            num = re.sub(r'\-$', '', e[1])
            word = mipinumreader.ReadNormalNumberWithBilingual(num, contextLanguage)
            if contextLanguage == 'chinese':
                word += ' 到 '
            else: 
                word += ' to '
        N = len(e)
        num = e[N-4] + e[N-3]
        word += mipinumreader.ReadNormalNumberWithBilingual(num, contextLanguage)
        if contextLanguage == 'chinese':
            if e[7] == '°C':
                word += ' 华 氏 度 '
            else:
                word += ' 摄 氏 度 '
        else:
            if e[7] == '°C':
                word += ' degree celsius '
            else: 
                word += ' degree fahrenheit '
        s = s.replace(pattern, ' {0} '.format(word))
        
        
        # m = re.findall(r'(((\-)?\d+(\.\d+)?\-)?(\-)?(\d+(\.\d+)?))(°C|°F)', s)
    return s
def GetLanguageContextByStrPair(sList, language):
    idx = 0
    wordList = list()
    if sList[0]:
        wordList = [ w for w in sList[0].split() if w ]
        idx = len(wordList)
    if len(sList) > 1 and sList[1]:
        list2 = sList[1].split()
        for x in list2:
            if x:
                wordList.append(x)
    return GetContextLanguage(wordList, idx, language)
def ReadEmail(s, language):
    m = re.findall(r'(\S+)@([^\.]+\.)(\S+\.)?(\S+)', s)
    for e in m:
        # logger.info('e={0}'.format(e))
        account =  ' at '
        pattern = e[0] + '@'
        idx = 1
        while idx < len(e):
            pattern += e[idx]
            if re.match(r'[^\.]+\.$', e[idx]):
                account += re.sub(r'([^\.]+)(\.)$', r' \1 dot ', e[idx])
            else:
                account +=  ' ' + e[idx]
            idx += 1
        # logger.info("pattern={0}, verbatim={1}".format(pattern, account))
        contextLanguage = GetLanguageContextByStrPair(s.split(pattern), language)
        userName = str(e[0])
        m1 = re.findall(r'\d+', userName)
        for x  in m1:
            pattern1 = str(x)
            verbatim1= ' '
            for d in list(x):
                verbatim1 += mipinumreader.ReadNormalNumberWithBilingual(d, contextLanguage) + ' '
            userName = userName.replace(pattern1, verbatim1)
        account = userName + account
        s = s.replace(pattern, ' {0} '.format(account))
    return s
def ReadBilingualYear(number, language):
    # if not IsYearNumber(number): return number
    if language == 'chinese':
        return ReadNumberBitByBit(number, language)
    elif language == 'english':
        return ReadYearByFourNumber(number)
    return number
def _ReadH1(wordList, idx, oovNormalizer, language):
    if not oovNormalizer.IsOovWord(wordList[idx]):
        return
    m = re.search(r'^([a-z]+)(\d+(\.\d+)?)$', wordList[idx])
    if not m: return
    word = m.group(1)
    number = m.group(2)
    # logger.info("word={0}, number={1}, hybrid_word={2}".format(word, number, wordList[idx]))
    contextLanguage = GetContextLanguage (wordList, idx, language)
    if re.match(r'^\d+\.\d+$', number):
        word +=  ' ' + mipinumreader.ReadNormalNumberWithBilingual(number, contextLanguage) 
    else:
        if int(number) < 100:
            word += ' ' +  mipinumreader.ReadNormalNumberWithBilingual(number, contextLanguage)
        else:
            if IsYearNumber(number):
                word += ' ' + ReadBilingualYear(number, contextLanguage)
            elif int(number) % 100 == 0:
                word += ' '+  mipinumreader.ReadNormalNumberWithBilingual(number, contextLanguage)
            else:
                for x in list(number):
                    word += ' ' +  mipinumreader.ReadNormalNumberWithBilingual(x, contextLanguage)
    wordList[idx] = word
def _ReadH2(wordList, idx, oovNormalizer, language):
    if not oovNormalizer.IsOovWord(wordList[idx]):
        return
    m = re.search(r'^(\d+(\.\d+)?)([^\d]+)$', wordList[idx])
    # logger.info("m={0}, group1={1}, group2={2}. group3={3}".format(m, m.group(1), m.group(2), m.group(3)))
    if not m: return
    number = m.group(1)
    word = m.group(3)
    if (word == 'st' or word == 'nd' or word == 'rd' or word == 'th') and IsInteger(number):
        wordList[idx] = _ReadOrdinal(int(number))
    elif word == 's':
        wordList[idx] = _ReadYearPluralSimple(number)
    else:
        contextLanguage = GetContextLanguage (wordList, idx, language)
        if word == 'bn': word = 'billion'
        wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(number, contextLanguage) + ' ' + word
def _ReadNormalBilingualData(wordList, idx, language):
    contextLanguage = GetContextLanguage(wordList, idx, language)
    if IsYearNumber(wordList[idx]):
        wordList[idx] = ReadBilingualYear(wordList[idx], contextLanguage)
    else:
        __ReadNormalData(wordList, idx, contextLanguage)
def _GetPercentSymbol(symbol, contextLanguage):
    if contextLanguage == 'english':
        if symbol == '%': 
            symbol = ' percent '
        else:
            symbol = ' per mille '
    else:
        if symbol == '%':
            symbol  = ' 百 分 之 '
        else:
            symbol = ' 千 分 之 '
    return symbol
def ReadOperator(oper, language):
    if oper == '+':
        if language == 'chinese': return '加'
        else: return 'plus'
    elif oper == '-':
        if language == 'chinese': return '减'
        else: return 'minus'
    elif oper == 'x':
        if language == 'chinese': return '乘'
        else: return 'times'
    elif oper == '/' or oper == '÷':
        if language == 'chinese': return '比'
        else: return 'is divided by'
    elif oper == '>':
        if language == 'chinese': return '大 于'
        else: return 'more than'
    elif oper == '<':
        if language == 'chinese': return '小 于'
        else: return 'less than'
    elif oper == '=':
        if language == 'chinese': return '等 于'
        else: return 'equals to'
    else:
        return ''
        # raise Exception("unknown operator: '{0}'".format(oper))
def __isOverallOperator(operList, oper):
    for op in operList:
        if op != oper:
            return False
    return True
def __IsChineseYearFormat(dataList):
    try:
        if len(dataList) == 3 and IsYearNumber(dataList[0]) and int(dataList[1]) <= 12 and int(dataList[2]) <=31:
            return True
    except ValueError:
        return False
    return False
def __IsWesternYearFormat(dataList):
    try:
        if len(dataList) == 3 and int(dataList[0]) <= 31 and int(dataList[1]) <= 12 and IsYearNumber(dataList[2]):
            return True
    except ValueError:
        return False
    return False
def __IsAllInteger(dataList):
    for x in dataList:
        if not IsInteger(x):
            return False
    return True
def __ReadFractionData(dataList, language):
    wordList = list()
    wordList.append(0)
    if len(dataList) > 2:
        if len(dataList) == 3 and __IsWesternYearFormat(dataList):
            t = dataList[0]
            dataList[0] = dataList[2]
            dataList[2] = t
            # logger.info("dataList={0}".format(dataList))
            _ReadEDashSDYear(dataList, wordList, 0, language)
            return wordList[0]
        else:
            word = str()
            if language == 'chinese':
                if __IsAllInteger(dataList):
                    word = ReadNumberBitByBit(dataList[0], 'chinese')
                    idx = 1
                    while idx < len(dataList):
                        word += ' 和 ' + ReadNumberBitByBit(dataList[idx], 'chinese')
                        idx += 1
                    return word
                else:
                    word = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'chinese')
                    idx = 1
                    while idx < len(dataList):
                        word += ' 比 ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[idx], 'chinese')
                        idx += 1
                    return word
            else:
                if __IsAllInteger(dataList):
                    word = ReadNumberBitByBit(dataList[0], 'english')
                    idx = 1
                    while idx < len(dataList):
                        word += ' and ' +  ReadNumberBitByBit(dataList[idx], 'english')
                        idx += 1
                    return word
                else:
                    word = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'english')
                    idx =1
                    while idx < len(dataList):
                         word += ' versus ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[idx], 'english')
                         idx += 1
                    return word
    else:
        ReadFractionData2(dataList[0], dataList[1], wordList, 0, language)
        return wordList[0]
            
def __ReadDashData(dataList, language):
    wordList = list()
    wordList.append(0)
    if language == 'english':
        _ReadEDashSD(dataList, wordList, 0)
    elif language == 'chinese':
         _ReadCDashSD(dataList, wordList, 0)
    else:
        raise Exception("unknown language: '{0}'".format(language))
    # logger.info("wordList={0}, dataList={1}, langauge={2}".format(wordList, dataList, language))
    return wordList[0]
def _ReadEquation(s, oovNormalizer, language):
    if re.search(r'(\d+(\.\d+)?)(([\+\-x\/÷]\d+(\.\d+)?){0,2})[\+\-x\/÷]\d+(\.\d+)?(([><=]\d+(\.\d+)?)?)',s):            
        m = re.findall(r'(\d+(\.\d+)?)(([\+\-x\/÷]\d+(\.\d+)?){0,2})([\+\-x\/÷]\d+(\.\d+)?(([><=]\d+(\.\d+)?)?))', s)
        for e in m:
            pattern = e[0] + e[2] + e[5]
            pattern = re.sub(r'\s+', '', pattern)
            sList = s.split(pattern)
            contextLanguage = GetLanguageContextByStrPair(sList, language)
            numList = re.split(r'[\+\-x\/÷<>=]', pattern)
            operList= re.sub(r'(\d+(\.\d+)?)', '', pattern)
            operList = list(operList)
            if len(operList) + 1 != len(numList):
                return s
                # raise Exception("illegal equation in utterance '{0}'".format(s))
            if __isOverallOperator(operList, '/'):
                # logger.info("numList={0}, pattern={1}".format(numList, pattern))
                word = __ReadFractionData(numList, contextLanguage)
                s = s.replace('{0}'.format(pattern), ' {0} '.format(word))
                # logger.info("utterance='{0}'".format(s))
                continue
            if __isOverallOperator(operList, '-'):
                word = __ReadDashData(numList, contextLanguage)
                s = s.replace('{0}'.format(pattern), ' {0} '.format(word))
                continue
            idx = 0
            word = mipinumreader.ReadNormalNumberWithBilingual(numList[idx], contextLanguage) 
            idx += 1
            while idx < len(numList):
                word += ' ' + ReadOperator(operList[idx-1], contextLanguage)
                word += ' ' + mipinumreader.ReadNormalNumberWithBilingual(numList[idx], contextLanguage) 
                idx += 1
            s1 = s
            s = s.replace('{0}'.format(pattern), ' {0} '.format(word))
            # logger.info("word={0}, s1={1}, s={2}, pattern={3}".format(word, s1, s, pattern))
    return s
def __IsTimeString(dataList):
    if __IsAllInteger(dataList) and len(dataList) >= 2:
        if len(dataList) == 2 and int(dataList[0]) < 24 and int(dataList[1]) <60:
            return True
        elif len(dataList) ==3 and int(dataList[2]) < 60:
            return True
    return False
def __ReadStringSequence(dataList, language, conSymbol):
    word = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], language)
    idx = 1
    while idx < len(dataList):
        word += ' {0} '.format(conSymbol) + mipinumreader.ReadNormalNumberWithBilingual(dataList[idx], language)
        idx += 1
    return word
def _ReadDataWithTimeFormat(s, oovNormalizer, language):
    if re.search(r'\d+(\.\d+)?((\s*:\d+(\.\d+)?){1,2})(\-\d+(\.\d+)?((\s*:\d+(\.\d+)?){1,2}))?', s):
        m = re.findall(r'(\d+(\.\d+)?((\s*:\d+(\.\d+)?){1,2}))(\-\d+(\.\d+)?((\s*:\d+(\.\d+)?){1,2}))?', s)
        for e in m:
            pattern = e[0] + e[5]
            # logger.info("e={0}, pattern={1}, utterance={2}".format(e, pattern, s))
            sList = s.split(pattern)
            contextLanguage = GetLanguageContextByStrPair(sList, language)
            sList = pattern.split('-')
            dataList = sList[0].split(':')
            wordList = list()
            wordList.append(0)
            if __IsTimeString(dataList):
                # logger.info("dataList={0}, sList={1}".format(dataList, sList))
                if contextLanguage == 'english':
                    _ReadECSD(dataList, wordList, 0)
                else:
                    _ReadCCSD(dataList, wordList, 0)
                word = wordList[0]
                if len(sList) == 2:
                    dataList = sList[1].split(':')
                    if __IsTimeString(dataList):
                        if contextLanguage == 'english':
                            word += ' to '
                            _ReadECSD(dataList, wordList, 0)
                            word += wordList[0]
                        else:
                            word += ' 到 '
                            _ReadCCSD(dataList, wordList, 0)
                            word += wordList[0]
                s = s.replace('{0}'.format(pattern), ' {0} '.format(word))
            elif len(sList) == 1:
                dataList = sList[0].split(':')
                word = str()
                if contextLanguage == 'english':
                    word = __ReadStringSequence(dataList, contextLanguage, 'versus')
                else:
                    word = __ReadStringSequence(dataList, contextLanguage, '比')
                s = s.replace('{0}'.format(pattern), ' {0} '.format(word))
    return s
def _ReadPercentOrMile(s, oovNormalizer, language):
    if re.search(r'(%|‰)', s):
        s = s.replace('~', '-')
        m = re.findall(r'(\d+(\.\d+\s*)?((%|‰){0,1})(\s*\-\s*\d+(\.\d+)?)?)(%|‰)', s)
        for e in m:
            pattern = e[0] + e[6]
            logger.debug("e={0}, pattern={1}, utterance={2}".format(e, pattern, s))
            word = re.sub(r'(%|‰|\s+)', '', e[0])
            sList = s.split(pattern)
            contextLanguage = GetLanguageContextByStrPair(sList, language)
            symbol = _GetPercentSymbol(e[6], contextLanguage)
            wordList = word.split('-')
            if len(wordList) == 1:
                if contextLanguage == 'english':
                    target = mipinumreader.ReadNormalNumberWithBilingual(wordList[0], contextLanguage) + ' ' + symbol
                    s = s.replace(pattern, ' {0} '.format(target))
                else:
                    target = symbol + ' ' + mipinumreader.ReadNormalNumberWithBilingual(wordList[0], contextLanguage)
                    s = s.replace(pattern, ' {0} '.format(target))
            elif len(wordList) == 2:
                if contextLanguage == 'english':
                    target = mipinumreader.ReadNormalNumberWithBilingual(wordList[0], contextLanguage) + ' ' + symbol + ' to ' + mipinumreader.ReadNormalNumberWithBilingual(wordList[1], contextLanguage) + ' ' + symbol
                    s = s.replace(pattern, ' {0} '.format(target))
                else:
                    target = symbol + ' ' + mipinumreader.ReadNormalNumberWithBilingual(wordList[0], contextLanguage) + ' 到 ' + symbol + ' ' + mipinumreader.ReadNormalNumberWithBilingual(wordList[1], contextLanguage)
                    s = s.replace(pattern, ' {0} '.format(target)) 
    return s
def ReadHybridWordWithNumber(s, oovNormalizer, language):
    
    s = _ReadPercentOrMile(s, oovNormalizer, language)
    s = _ReadDataWithTimeFormat(s, oovNormalizer, language)
    s = _ReadEquation(s, oovNormalizer, language)
    s = __ReadEnglishYearInterval(s)

    wordList= [ w  for w in s.split() if w.strip() ]
    length = len(wordList)
    idx = 0
    while idx < length:
        if re.match(r'^[a-z]+\d+(\.\d+)?$', wordList[idx]):
            _ReadH1(wordList, idx, oovNormalizer, language)
        elif re.match(r'^\d+(\.\d+)?[^\d]+$', wordList[idx]):
            _ReadH2(wordList, idx, oovNormalizer, language)
        elif re.match(r'^\d+(\.\d+)?$', wordList[idx]):
            _ReadNormalBilingualData(wordList, idx, language)
        idx += 1
    # sys.stderr.write("wordList={0}\n".format(wordList))
    return ' '.join(str(v) for v in wordList)
def PreprocessUtterance(s, oovNormalizer, language):
    # s = ConvertT2SChinese(s)
    s = SegmentChineseUtterance(s)
    # logger.info("line={0}".format(s))
    s = RemovePunctuation(s)
    s = s.lower()
    # s = ConvertEnglishNumberToArabicNumber(s)
    s = re.sub(r'(\d+(\.\d+)?)\s*%', r'\1% ', s)
    s = re.sub(r'(\d+(\.\d+)?)\s*‰', r'\1‰ ', s)
    s = NormalizeNormalNumber(s)
    s = NormalizePercentNumber(s)
    s = NormalizeCurrencyNumber(s)
    s = ReadTemperature(s, language)
    s = ReadEmail(s, language)
    s = ReadHybridWordWithNumber(s, oovNormalizer, language)
    s = re.sub(r'(www)\.(\S+)\.(\S+)\.(\S+)', r'\1 dot \2 dot \3 dot \4', s)
    s = re.sub(r'(www)\.(\S+)\.(\S+)', r'\1 dot \2 dot \3', s)
    s = re.sub(r'[\(;\)\[\]]', ' ', s)
    # s = re.sub(r'@\s*(\S+)\.(\S+)', r' at \1 dot \2 ', s).lower()
    s = NormalizeWhiteSpace(s)
    s = __PostprocessDataInUtterance(s, oovNormalizer, language)
    s = ConvertChineseData(s)
    return s
def CountChineseEnglishWord(wordList):
    countList = [0, 0, 0, 0]
    countList[0] = len(wordList)
    for w in wordList:
        if IsEnglishWord(w): countList[1] += 1
        if re.match(r'\d+(\.\d+)?\s*%', w): countList[3] += 1
        if re.match(r'(€|\$|£)\d+(\.\d+)?', w): countList[3] += 1
        # if re.match(r'£\d+(\.\d+)?', w): countList[3] += 1
        if re.match(r'(\d+\s*\/\s*\d+)', w): countList[3] += 1
        if re.match(r'(\d+\s*:\s*\d+)', w): countList[3] += 1
        if re.match(r'(\d+(\-\d+)?\-\d+)', w): countList[3] += 1
        if re.match(r'(\d+:\d+\-\d+:\d+)', w): countList[3] += 1
        if re.match(r'^(\-)?\d+(\.\d+)?\-(\-)?\d+(\.\d+)?$', w): countList[3] += 1
        if re.match(r'(\d+\s*\+\s*\d+)', w): countList[3] += 1
        if re.match(r'(\d+(\.\d+)?\s*x\s*\d+(\.\d+)?)', w): countList[3] += 1
        if re.match(r'\d+(st|nd|rd|th)', w): countList[3] += 1
        if re.match(r'(\d+0s\s*(\-\s*\d+0s)?)', w): countList[3] += 1
        if re.match(r'<s>', w): countList[3] += 1
        if IsRealNumber(w): countList[3] += 1
        if IsChineseWord(w): countList[2] += 1
    return countList
def GetLeftLanguageContext(idx, wordList):
    leftContext = str()
    while idx >= 0:
        if IsEnglishWord(wordList[idx]):
            leftContext = 'english'
            break
        if IsChineseWord(wordList[idx]):
            leftContext = 'chinese'
            break
        idx -= 1
    return leftContext
def GetRightLanguageContext(idx, wordList):
    length = len(wordList)
    rightContext = str()
    while idx < length:
        if IsEnglishWord(wordList[idx]):
            rightContext = 'english'
            break
        if IsChineseWord(wordList[idx]):
            rightContext = 'chinese'
            break
        idx += 1
    return rightContext
def _ReadEPD(number, wordList, idx):
    wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(number, 'english') + ' percent'
    return
def _ReadCPD(number, wordList, idx):
    wordList[idx] = '百 分 之 ' + mipinumreader.ReadNormalNumberWithBilingual(number, 'chinese')
    return
def ReadPercentData(number, wordList, idx, language):
    contextLanguage = GetContextLanguage (wordList, idx, language)
    length = len(wordList)
    if contextLanguage == 'english':
        _ReadEPD(number, wordList, idx)
    elif contextLanguage == 'chinese':
        _ReadCPD(number, wordList, idx)
    else:
        raise Exception("unknown context language '{0}'".format(contextLanguage))
def ReadCurrencyData(currency, number, wordList, idx, language):
    contextLanguage = GetContextLanguage (wordList, idx, language)
    word = str()
    if contextLanguage == 'english':
        word = mipinumreader.ReadNormalNumberWithBilingual(number, 'english')
        if currency == '$': 
            word = word + ' dollars '
        elif currency == '£':
            word = word + ' pounds '
        elif currency == '€':
            word = word + ' euros '
    elif contextLanguage == 'chinese':
        word = mipinumreader.ReadNormalNumberWithBilingual(number, 'chinese')
        if currency == '$':
            word = word + ' 美 元 '
        elif currency == '£':
            word = word + ' 英 镑 '
        elif currency == '€':
            word = word + ' 欧 元 '
    if word:
        wordList[idx] = word
def ReadNumberBitByBit(num, language):
    numList = list(str(num))
    s = str()
    for number in numList:
        s += mipinumreader.ReadNormalNumberWithBilingual(number, language) + ' '
    return s.strip()
def _ReadEFD(num, den, wordList, idx):
    if float(num) >= float(den) or len(str(num)) >= 4:
        if len(str(num)) <= 4 and len(str(num)) <= 4:
            wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(num, 'english') + ' versus ' + mipinumreader.ReadNormalNumberWithBilingual(den, 'english')
        else:
            if __IsAllInteger([num, den]):
                wordList[idx] = ReadNumberBitByBit(num, 'english') + ' and ' + ReadNumberBitByBit(den, 'english')
            else:
                wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(num, 'english') + ' versus ' + mipinumreader.ReadNormalNumberWithBilingual(den, 'english')
    else:
        if __IsAllInteger([num, den]):
            word = str()
            num = int(num)
            den = int(den)
            if num == 1 and den == 2: word = 'half'
            elif num == 9 and den == 11: word = 'nine eleven'
            elif num == 1: word = str(_ReadPostiveInteger(num)) + ' ' + str(_ReadOrdinal(den))
            elif num > 1 and num < 10: word = str(_ReadPostiveInteger(num)) + ' ' + str(_ReadOrdinal(den)) + 's' 
            else:
                word = str(_ReadPostiveInteger(num)) + ' versus ' + str(_ReadPostiveInteger(den))
            wordList[idx] = word
        else:
            wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(num, 'english') + ' versus ' + mipinumreader.ReadNormalNumberWithBilingual(den, 'english')
def _ReadCFD(num, den, wordList, idx):
    if float(num) >= float(den) or len(str(num)) >= 4:
        if len(str(num)) < 4 and len(str(num)) < 4:
            wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(num, 'chinese') + ' 比 ' + mipinumreader.ReadNormalNumberWithBilingual(den, 'chinese')
        else:
            if __IsAllInteger([num, den]):
                wordList[idx] = ReadNumberBitByBit(num, 'chinese') + ' 和 ' + ReadNumberBitByBit(den, 'chinese')
            else:
                wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(num, 'chinese') + ' 比 ' + mipinumreader.ReadNormalNumberWithBilingual(den, 'chinese')
    else:
        word = str()
        if __IsAllInteger([num, den]) and int(num) == 9 and int(den) == 11: word = ' 九 幺 幺'
        else:
            if float(num) >= 10 or not __IsAllInteger([num, den]):
                word = mipinumreader.ReadNormalNumberWithBilingual(num, 'chinese') + ' 比 ' + mipinumreader.ReadNormalNumberWithBilingual(den, 'chinese')
            else:
                word = mipinumreader.ReadNormalNumberWithBilingual(den, 'chinese') + ' 分 之 ' + mipinumreader.ReadNormalNumberWithBilingual(num, 'chinese')
        wordList[idx] = word
        
def ReadFractionData2(num, den, wordList, idx, language):
    contextLanguage = GetContextLanguage(wordList, idx, language)
    if contextLanguage == 'english':
        _ReadEFD(num, den, wordList, idx)
    elif contextLanguage == 'chinese':
        _ReadCFD(num, den, wordList, idx)
    else:
        raise Exception("unknown contextual language '{0}'".format(contextLanguage))
def _ReadECSD(dataList, wordList, idx):
    dataSize = len(dataList)
    if dataSize < 2: return
    if int(dataList[0]) < 24 and int(dataList[1]) < 60:
        dataList[0] = re.sub(r'^0+', '', dataList[0])
        hour = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'english')
        minute = ''
        if int(dataList[1]) > 0:
            minute = mipinumreader.ReadNormalNumberWithBilingual(dataList[1], 'english')
        if int(dataList[1]) < 10:
            if int(dataList[1]) == 0:
                hour += ' o\'clock'
            else:
                minute = 'past ' + minute
        second = ''
        if dataSize == 3 and int(dataList[2]) > 0 and int(dataList[2]) < 60:
            second = 'and ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[2], 'english')  + ' seconds'
        wordList[idx] = hour + ' ' + minute + ' ' + second
    else:
        wordList[idx] = ''
        for data in dataList:
            wordList[idx] += mipinumreader.ReadNormalNumberWithBilingual(data, 'english') + ' '
def _ReadCCSD(dataList, wordList, idx):
    dataSize = len(dataList)
    if int(dataList[0]) < 24 and int(dataList[1]) < 60:
        dataList[0] = re.sub(r'^0+', '', dataList[0])
        hour  = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'chinese') + ' 点'
        minute = ''
        if int(dataList[1]) > 0:
            minute = mipinumreader.ReadNormalNumberWithBilingual(dataList[1], 'chinese') + ' 分'
        second = ''
        if dataSize == 3 and int(dataList[2]) > 0 and int(dataList[2]) < 60:
            second = mipinumreader.ReadNormalNumberWithBilingual(dataList[2], 'chinese')  + ' 秒'
            if not minute: minute = ' 零 '
        wordList[idx] = hour + ' ' + minute + ' ' + second
    else:
        wordList[idx] = ''
        for data in dataList:
            wordList[idx] += mipinumreader.ReadNormalNumberWithBilingual(data, 'chinese') + ' 比 '
        wordList[idx] = re.sub(r' 比$', '', wordList[idx])
def GetContextLanguage(wordList, idx, language):
    if language == 'english':
        return 'english'
    elif language == 'chinese':
        return 'chinese'
    else:
        if idx > 0:
            leftContext = GetLeftLanguageContext(idx-1, wordList)
            if leftContext == 'english':
                return 'english'
            elif leftContext == 'chinese':
                return 'chinese'
            else:
                rightContext = GetRightLanguageContext(idx+1, wordList)
                if rightContext == 'english':
                    return 'english'
                else:
                    return 'chinese'
        else:
            rightContext = GetRightLanguageContext(idx+1, wordList)
            if rightContext =='english':
                return 'english'
            else:
                return 'chinese'
def ReadColonStringData(dataList, wordList, idx, language):
    contextLanguage = GetContextLanguage(wordList, idx, language)
    if contextLanguage == 'english':
        _ReadECSD(dataList, wordList, idx)
    elif contextLanguage == 'chinese':
        _ReadCCSD(dataList, wordList, idx)
    else:
        raise Exception("unknown contextual language '{0}'".format(contextLanguage))
def _ReadEnglishMonth(month):
    monthDict = {1:'january', 2:'february', 3:'march', 4:'april', 5:'may', 6:'june', 7:'july', 8:'august', 9:'september', 10:'october', 11:'november', 12:'december'}
    if int(month) in monthDict:
        return monthDict[int(month)]
    return month
def _ReadEDashSDYear(dataList, wordList, idx, language):
    if language == 'english':
        if len(dataList)== 3 and IsYearNumber(dataList[0]) and int(dataList[1]) <= 12 and int(dataList[2]) <= 31:
            year = ReadYearByFourNumber(dataList[0])
            month = _ReadEnglishMonth(dataList[1])
            day = _ReadOrdinal(int(dataList[2]))
            # sys.stderr.write("year={0}, month={1}, day={2}\n".format(year, month, day))
            wordList[idx] = month + ' ' + day + ' ' + year
        elif len(dataList) == 2 and IsYearNumber(dataList[0]) and IsYearNumber(dataList[1]) and int(dataList[1]) > int(dataList[0]):
            wordList[idx] = ReadYearByFourNumber(dataList[0]) + ' to ' + ReadYearByFourNumber(dataList[1])
        elif len(dataList) == 2 and IsYearNumber(dataList[0]) and len(str(dataList[1])) == 2 and int(float(dataList[0]) / 100)*100  + int(dataList[1]) > int(dataList[0]):
            wordList[idx] = ReadYearByFourNumber(dataList[0]) + ' to '  + mipinumreader.ReadNormalNumberWithBilingual(dataList[1], 'english')
        else:
            wordList[idx] = str()
            for x in dataList:
                wordList[idx]  += ReadNumberBitByBit(x, 'english') + ' '
    else:
        if len(dataList)== 3 and IsYearNumber(dataList[0]) and int(dataList[1]) <= 12 and int(dataList[2]) <= 31:
            dataList[1] = re.sub(r'^0', '', dataList[1])
            month =  mipinumreader.ReadNormalNumberWithBilingual(dataList[1], 'chinese')
            month = re.sub(r'^一', '', month)
            wordList[idx] = ReadNumberBitByBit(dataList[0], 'chinese') + ' 年 ' + month + ' 月 ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[2], 'chinese') + ' 日 '
        elif len(dataList) == 2 and IsYearNumber(dataList[0]) and IsYearNumber(dataList[1]) and int(dataList[1]) > int(dataList[0]):
            wordList[idx] = ReadNumberBitByBit(dataList[0], 'chinese') + ' 到 ' + ReadNumberBitByBit(dataList[1], 'chinese')
        elif len(dataList) == 2 and IsYearNumber(dataList[0]) and len(str(dataList[1])) == 2 and int(float(dataList[0]) / 100)*100  + int(dataList[1]) > int(dataList[0]):
            wordList[idx] = ReadNumberBitByBit(dataList[0], 'chinese') + ' 到 ' + ReadNumberBitByBit(dataList[1], 'chinese') 
        else:
            wordList[idx] = str()
            for x in dataList:
                wordList[idx]  += ReadNumberBitByBit(x, 'chinese') + ' and '
def IsYearInterpretable(dataList):
    if __IsWesternYearFormat(dataList):
        return True
    if __IsChineseYearFormat(dataList):
        return True
    return False
def IsYearInterval(dataList):
    try:
        if len(dataList) == 2 and int(dataList[0]) < int(dataList[1]) and IsYearNumber(dataList[0]) and IsYearNumber(dataList[1]):
            return True
    except ValueError:
        return False
    return False
def __isIncrementalOrder(dataList):
    length = len(dataList)
    if length == 1: return False
    x1 = dataList[0]
    idx = 1
    while idx < length:
        x2 = dataList[idx]
        if x1 >= x2:
            return False
        x1 = x2
        idx += 1
    return True
def __IsAllInteger(dataList):
    if not dataList: return False
    for x in dataList:
        if not IsInteger(x):
            return False
    return True
def _ReadEDashSD(dataList, wordList, idx):
    # sys.stderr.write("dataList={0}\n".format(dataList))
    if IsYearInterpretable(dataList):
        _ReadEDashSDYear(dataList, wordList, idx, 'english')
        return
    if IsYearInterval(dataList):
        wordList[idx] = ReadYearByFourNumber(dataList[0]) + ' to ' + ReadYearByFourNumber(dataList[1])
        return
    if __isIncrementalOrder(dataList) and float(dataList[len(dataList)-1]) < 10000:
        wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'english')
        i = 1
        while i < len(dataList):
            wordList[idx] += ' to ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[i], 'english')
            i += 1
    else:
        if __IsAllInteger(dataList):
            wordList[idx] = ReadNumberBitByBit(dataList[0], 'english')
            i = 1
            while i < len(dataList):
                wordList[idx]  += ' and ' + ReadNumberBitByBit(dataList[i], 'english')
                i += 1
        else:
            wordList[idx] = '-'.join(dataList)
        
def _ReadCDashSD(dataList, wordList, idx):
    if IsYearInterpretable(dataList):
        _ReadEDashSDYear(dataList, wordList, idx, 'chinese')
    elif IsYearInterval(dataList):
        wordList[idx] = ReadNumberBitByBit(dataList[0], 'chinese') + ' 到 ' + ReadNumberBitByBit(dataList[1], 'chinese') + ' 年 '
    elif __isIncrementalOrder(dataList)  and float(dataList[len(dataList)-1]) < 10000:
        wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'chinese')
        i = 1
        while i < len(dataList):
            wordList[idx] += ' 到 ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[i], 'chinese')
            i += 1
    else:
        if __IsAllInteger(dataList):
            wordList[idx] = ReadNumberBitByBit(dataList[0], 'chinese')
            i = 1
            while i < len(dataList):
                wordList[idx]  +=  ' 杠 ' + ReadNumberBitByBit(dataList[i], 'chinese')
                i += 1
        else:
            wordList[idx] = '-'.join(dataList)

def ReadDashStringData(dataList, wordList, idx, language):
    contextLanguage = GetContextLanguage(wordList, idx, language)
    if contextLanguage == 'english':
        _ReadEDashSD(dataList, wordList, idx)
    elif contextLanguage == 'chinese':
        _ReadCDashSD(dataList, wordList, idx)
    else:
        raise Exception ("unknown context language '{0}'".format(contextLanguage))
def _ReadEPOTF(operList, dataList, wordList, idx):
    if len(operList) + 1 != len(dataList):
        raise Exception("mismatch betweenn length of operList '{0}' and dataList '{1}'".format(len(operList), len(dataList)))
    wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'english')
    # sys.stderr.write("wordList={0}\n".format(dataList[0]))
    i = 0
    while i < len(operList):
        if (operList[i] == '+'):
            wordList[idx] += ' plus'
        elif operList[i] == 'x':
            wordList[idx] += ' times'
        else:
            raise Exception("unidentified operator '{0}'".format(operList[i]))
        wordList[idx] += ' ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[i+1], 'english')
        i += 1
def _ReadCPOTF(operList, dataList, wordList, idx):
    if len(operList) + 1 != len(dataList):
        raise Exception("mismatch betweenn length of operList '{0}' and dataList '{1}'".format(len(operList), len(dataList)))
    wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'english')
    i = 0
    while i < len(operList):
        if (operList[i] == '+'):
            wordList[idx] += ' plus'
        elif operList[i] == 'x':
            wordList[idx] += ' times'
        else:
            raise Exception("unidentified operator '{0}'".format(operList[i]))
        wordList[idx] += ' ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[i+1], 'english')
        i += 1
def _ReadCPOTF(operList, dataList, wordList, idx):
    if len(operList) + 1 != len(dataList):
        raise Exception("mismatch betweenn length of operList '{0}' and dataList '{1}'".format(len(operList), len(dataList)))
    wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(dataList[0], 'chinese')
    i = 0
    while i < len(operList):
        if (operList[i] == '+'):
            wordList[idx] += ' 加'
        elif operList[i] == 'x':
            wordList[idx] += ' 乘'
        else:
            raise Exception("unidentified operator '{0}'".format(operList[i]))
        wordList[idx] += ' ' + mipinumreader.ReadNormalNumberWithBilingual(dataList[i+1], 'chinese')
        i += 1
def ReadPlusOrTimeFormula(operList, dataList, wordList, idx, language):
    contextLanguage = GetContextLanguage(wordList, idx, language)
    if contextLanguage == 'english':
        _ReadEPOTF(operList, dataList, wordList, idx)
    elif contextLanguage == 'chinese':
        _ReadCPOTF(operList,dataList, wordList, idx)
    else:
        raise Exception("unknown contextual language '{0}'".format(contextLanguage))
def ReadTimeInterval(wordList, idx, language):
    mList = wordList[idx].split('-')
    dataList = [ int(w) for w in mList[0].split(':') ]
    contextLanguage = GetContextLanguage(wordList, idx, language)
    ReadColonStringData(dataList, wordList, idx, contextLanguage)
    word = wordList[idx]
    if contextLanguage == 'chinese':
        word = wordList[idx] + ' 到 '
    else:
        word = wordList[idx] + ' to '
    wordList[idx] = str()
    dataList = [ int(w) for w in mList[1].split(':') ]
    ReadColonStringData(dataList, wordList, idx, contextLanguage)
    word += wordList[idx]
    wordList[idx] = word
def __ReadNormalData(wordList, idx, language):
    contextLanguage = GetContextLanguage(wordList, idx, language)
    if contextLanguage == 'english':
        wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(wordList[idx], 'english')
    elif contextLanguage == 'chinese':
        wordList[idx] = mipinumreader.ReadNormalNumberWithBilingual(wordList[idx], 'chinese')
    else:
        raise Exception("unknown contextual language '{0}'".format(contextLanguage))
""" The following function is to be removed, since we already do those processing in the PreprocessUtterance function
"""
def NormalizeWordWithLinguisticContext(lingCountList, wordList, idx, args):
    if lingCountList[3] == 0:
        return
    if re.match(r'(\d+(\.\d+)?)\s*%', wordList[idx]):
        m = re.match(r'(\d+(\.\d+)?)\s*%', wordList[idx])
        ReadPercentData(m.group(1), wordList, idx, args.language.lower())
        return
    if re.match(r'(\$|£|€)(\d+(\.\d+)?)', wordList[idx]):
        m = re.match(r'(\$|£|€)(\d+(\.\d+)?)', wordList[idx])
        ReadCurrencyData(m.group(1), m.group(2), wordList, idx, args.language.lower())
        return
    if re.match(r'(\d+\s*\/\s*\d+)', wordList[idx]):
        m = re.match(r'(\d+)\s*\/\s*(\d+)', wordList[idx])
        ReadFractionData2(int(m.group(1)), int(m.group(2)), wordList, idx, args.language.lower())
        return
    if re.match(r'(\d+\s*(:\s*\d+)?:\s*\d+)$', wordList[idx]):
        mList = [ int(w) for w in wordList[idx].split(':') ]
        ReadColonStringData(mList, wordList, idx, args.language.lower())
        return
    if re.match(r'^((\-)?\d+(\.\d+)?)\-((\-)?\d+(\.\d+)?)$', wordList[idx]):
        m = re.match(r'^((\-)?\d+(\.\d+)?)\-((\-)?\d+(\.\d+)?)$', wordList[idx])
        mList = list()
        mList.append(m.group(1))
        mList.append(m.group(4))
        ReadDashStringData(mList, wordList, idx, args.language.lower())
        return
    if re.match(r'(\d+(\.\d+)?\s*(\-\s*\d+(\.\d+)?)?\-\s*\d+(\.\d+)?)$', wordList[idx]):
        # sys.stderr.write("word={0}\n".format(wordList[idx]))
        mList = [w for w in wordList[idx].split('-')]
        ReadDashStringData(mList, wordList, idx, args.language.lower())
        return
    if re.match(r'\d+(\.\d+)?((\+|x)\d+(\.\d+)?)?(\+|x)\s*\d+(\.\d+)?', wordList[idx]):
        operList= re.findall(r'(\+|x)', wordList[idx])
        dataList = re.split(r'\+|x', wordList[idx])
        # sys.stderr.write("word={0}\n".format(wordList[idx]))
        if len(operList) + 1 != len(dataList): return
        ReadPlusOrTimeFormula(operList, dataList, wordList, idx, args.language.lower())
        return
    elif re.match(r'^(\d+:\d+\-\d+:\d+)$', wordList[idx]):
        ReadTimeInterval(wordList, idx, args.language.lower())
    elif re.match(r'^\d+(\.\d+)?$', wordList[idx]):
        logger.info("normal number is '{0}'".format(wordList[idx]))
        if IsYearNumber(wordList[idx]):
            contextLanguage = GetContextLanguage(wordList, idx, args.language.lower())
            wordList[idx] = ReadBilingualYear(wordList[idx], contextLanguage)
        else:
            __ReadNormalData(wordList, idx, args.language.lower())
        
    return
def NormalizeTextWithLinguisticContext(s, args):
    wordList = [w.lower() for w in s.split() if w.strip()]
    length = len(wordList)
    idx = 0
    countList = CountChineseEnglishWord(wordList)
    if countList[0] < 1: 
        return str()
    elif (countList[1] + countList[2] + countList[3])*100/countList[0] < 80:
        # logger.info("utterance={0}".format(s))
        return str()
    elif countList[3] == 0:
        return s
    while idx < length:
        if IsChineseWord(wordList[idx]):
            idx += 1
        elif IsEnglishWord(wordList[idx]):
            idx += 1
        else:
            # sys.stderr.write("s={0}\n".format(s))
            NormalizeWordWithLinguisticContext(countList, wordList, idx, args)
            idx += 1
    return ' '.join(wordList)
def RemoveLineWithTooManyOov(line, oovNormalizer, scaleFactor):
    wordList = [ w for w in line.split() if w.strip() ]
    totalNumOfWord = len(wordList)
    numOfInVocab = 0
    for w in wordList:
        if not oovNormalizer.IsOovWord(w):
            numOfInVocab += 1
    if numOfInVocab*scaleFactor <= totalNumOfWord:
        return str()
    return line
def RemoveDuplicatedLine(line, uniqueLineDict):
    if line in uniqueLineDict:
        uniqueLineDict[line] += 1
        return str()
    uniqueLineDict[line] = int(1)
    return line
def NormalizeText(args):
    oovNormalizer = OovNormalizer(args)
    oovNormalizer.LoadDicts()
    uniqueLineDict = dict()
    if args.text_file:
        global ostream
        if args.tgtdir:
            ostream = open("{0}".format(args.tgtdir), 'w', encoding='utf-8')
        lineNum = 0
        with codecs.open(args.text_file, encoding='utf-8', errors='ignore') as istream:
            for line in istream:
                line = line.strip()
                if args.language.lower() == "english":
                    line =  NormalizeEnglishUtterance(line, oovNormalizer)
                if not line: continue
                oldLine = line
                line  = PreprocessUtterance(line, oovNormalizer, args.language.lower())

                line = RemoveLineWithTooManyOov(line, oovNormalizer, 2.0)

                if not line:
                    logger.debug("'{0}' is removed due to too many oov words".format(oldLine))
                    continue
                line = RemoveDuplicatedLine(line, uniqueLineDict)
                if not line:
                    logger.debug("'{0}' is removed due to duplication".format(oldLine))
                    continue
                line = NormalizeTextWithLinguisticContext(line, args)
                if not line:
                    logger.debug("'{0}' is removed\n".format(oldLine))
                    continue
                if args.tgtdir and line:
                    strList = PostprocessUtterance(line, oovNormalizer, args.language.lower(), True)

                    for utterance in strList:
                        if utterance:
                            ostream.write("{0}\n".format(utterance.strip()))
                    # sys.stderr.write("oldLine={0},\nline={1}\n".format(oldLine, line))
                lineNum += 1
                if(lineNum%1000 ==0):
                    logger.info("Processed line {0}".format(lineNum))
        if args.tgtdir:
            ostream.close()
            logger.info("Save the normalized text in '{0}'".format(args.tgtdir))
"""  
 normalize kaldi format text, do not break the line
"""
def NormalizeTextWithUttId(args):
    oovNormalizer = OovNormalizer(args)
    oovNormalizer.LoadDicts()
    uniqueLineDict = dict()
    if args.text_file:
        global ostream
        if args.tgtdir:
            ostream = open("{0}".format(args.tgtdir), 'w', encoding='utf-8')
        lineNum = 0
        with codecs.open(args.text_file, encoding='utf-8', errors='ignore') as istream:
            for line in istream:
                line = line.strip()
                wordList = line.split();
                uttid = wordList.pop(0)
                # if not re.match(r'^\d+\-\d+$', uttid):
                #    raise Exception("unknown utterance id:'{0}'".format(uttid))
                line = ' '.join(wordList).strip()  # normalize the text
                line = ProcessYearUtterance(line, args.language.lower())

                line = ProcessChinesesArabicYear(line)

                if args.language.lower() == "english":
                    line =  NormalizeEnglishUtterance(line, oovNormalizer)
                if not line: continue
                oldLine = line
                line  = PreprocessUtterance(line, oovNormalizer, args.language.lower())
                # logger.info("line={0}".format(line))
                # line = RemoveLineWithTooManyOov(line, oovNormalizer, 2.0)

                if not line:
                    logger.debug("'{0}' is removed due to too many oov words".format(oldLine))
                    continue
                # line = RemoveDuplicatedLine(line, uniqueLineDict)
                if not line:
                    logger.debug("'{0}' is removed due to duplication".format(oldLine))
                    continue
                line = NormalizeTextWithLinguisticContext(line, args)
                if not line:
                    logger.debug("'{0}' is removed\n".format(oldLine))
                    continue
                if args.tgtdir and line:
                    strList = PostprocessUtterance(line, oovNormalizer, args.language.lower(), False)
                    utterance = ' , '.join(strList).strip()
                    ostream.write("{0} {1}\n".format(uttid, utterance))
                    # sys.stderr.write("oldLine={0},\nline={1}\n".format(oldLine, line))
                lineNum += 1
                if(lineNum%1000 ==0):
                    logger.info("Processed line {0}".format(lineNum))
        if args.tgtdir:
            ostream.close()
            logger.info("Save the normalized text in '{0}'".format(args.tgtdir))
"""
for id in $(diff ./blizzard/blizzard_release_2019_v1/april-23-wave-concatenation/eval/redo-normalized_cut_mc_text.txt ./blizzard/blizzard_release_2019_v1/april-23-wave-concatenation/eval/normalized_cut_mc_text.txt | perl -ane 'use utf8; if(/^>\s+(\S+)/){ print "$1\n";  }  ' ); do grep $id ./blizzard/blizzard_release_2019_v1/april-23-wave-concatenation/eval/cut_mc_text.txt; done
"""
def ProcessYearUtterance(s, language):
    if re.search(r'\d+\s*年', s):
        findList = re.findall(r'(\d+)\s*年', s)
        for year in findList:
            pattern = year
            to = str();
            if re.search(r'-{0}'.format(year), s):
                pattern = '-{0}'.format(year)
                to = "到"
            elif re.search(r'－{0}'.format(year), s):
                pattern = '－{0}'.format(year)
                to = "到"
            year_str = ReadBilingualYear(year, "chinese")
            # logger.info("year={0}, year_str={1}".format(year, year_str))
            target_pattern = "{0} {1}".format(to, year_str)
            s = s.replace(pattern, target_pattern)
    return s
def ProcessChinesesArabicYear(s):
    if re.search(r'(\d{4}|\d{2})\.\d+\.\d+', s):
        findList = re.findall(r'((\d{4}|\d{2})\.\d+\.\d+)', s)
        for mlist in findList:
            pattern = mlist[0]
            time_list = pattern.split('.')
            if int(time_list[1]) <= 12 and int(time_list[2]) <= 31:
                target = ReadBilingualYear(time_list[0], 'chinese') + ' 年 '
                target = target + mipinumreader.ReadNormalNumberWithBilingual(time_list[1], 'chinese') + ' 月 '
                target = target + mipinumreader.ReadNormalNumberWithBilingual(time_list[1], 'chinese') + ' 日 '
                s = s.replace(pattern, target)
    return s


def __PostprocessDataInUtterance(s, oovNormalizer, language):
    if re.search(r'(\-|\+)?\d+(\.\d+)?', s):
        findList = re.findall(r'((\-|\+)?\d+(\.\d+)?)', s)
        for m in findList:
            pattern = m[0]
            if pattern[0] == '+':  # we cannot process such a pattern
                return s
            verbatim = str()
            contextLanguage = GetLanguageContextByStrPair(s.split(pattern), language)
            ## logger.info("pattern={0}, language={1}".format(pattern, language))
            if re.search(r'{0}\s*(年|米|千)'.format(pattern), s):
                if pattern[0] == '-':
                    content = "{}".format(pattern[1:])
                    # logger.info("pattern={0}".format(content))
                    # if IsYearNumber(content):
                    if re.search(r'{0}\s*年'.format(pattern), s):
                        verbatim =  ReadBilingualYear(content, contextLanguage)
                    else:
                        verbatim = mipinumreader.ReadNormalNumberWithBilingual(content, contextLanguage) 
                    verbatim = "到 {0}".format(verbatim)
                else:
                    
                    if re.search(r'{0}\s*年'.format(pattern), s):
                        verbatim =  ReadBilingualYear(pattern, contextLanguage)
                    else:
                        verbatim = mipinumreader.ReadNormalNumberWithBilingual(pattern, contextLanguage) 
            else:
                verbatim = mipinumreader.ReadNormalNumberWithBilingual(pattern, contextLanguage)
            # logger.info("pattern={0}, contextLanguage={1}, verbatim={2}, s ={3}".format(pattern, contextLanguage, verbatim, s))
            s = s.replace(pattern, " {0} ".format(verbatim))
                # logger.info("s={0}".format(s))
    return s
def PostprocessUtterance(s, oovNormalizer, language, do_removal):
    s = re.sub(r'\-|”|“|\/', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    strList = re.split(r'<s>|\||。|，|\.|,|\?|—|——', s)
    newList = list()
    for line in strList:
        oldLine = line
        if not line: continue
        if do_removal:
            line = RemoveLineWithTooManyOov(line, oovNormalizer, 1.5)
        if line and re.search(r'\d+(\.\d+)?', line):
            line = re.sub('@', ' at ', line)
            line = ReadHybridWordWithNumber(line, oovNormalizer, language) # bad, but works
            line = __PostprocessDataInUtterance(s, oovNormalizer, language)
        if line.strip():
            newList.append(line)
        else:
            logger.info("'{0}' removed with postprocessing".format(oldLine))

    return newList
def TransferMipitalkMobileData(args):
    oovNormalizer =OovNormalizer(args)
    oovNormalizer.LoadDicts()
    outputLines = str()
    arabicLex = dict()
    if args.text_file:
        with open(args.text_file, encoding='utf-8') as istream:
            for line in istream:
                retList = oovNormalizer.GetLabelAndWords(line, True)
                if not retList: continue
                label = retList[0]
                words = retList[1]
                words = oovNormalizer.TransferWords(words)
                outputLines += label + ' ' + words + "\n"
                for word in words.split():
                    if not word: continue
                    if re.match(r'(\d+(\.\d+)?)', word):
                        m = re.match(r'(\d+(\.\d+)?)', word)
                        curword = m.group(1)
                        # logger.info("curword={0}".format(curword))
                        oovNormalizer.WriteArabicNumberLexicon(word, arabicLex)
                    elif oovNormalizer.IsOovWord(word):
                        oovNormalizer.AddWordToOovDict(word)
    if args.tgtdir:
        ostream = open("{0}/text".format(args.tgtdir), 'w', encoding='utf-8')
        ostream.write(outputLines)
        ostream.close()
        oovStrs = oovNormalizer.DumpOovDict()
        ostream = open("{0}/oov.txt".format(args.tgtdir), 'w', encoding='utf-8')
        ostream.write(oovStrs)
        ostream.close()
        ostream = open("{0}/arabic-lexicon.txt".format(args.tgtdir), 'w', encoding='utf-8')
        s = oovNormalizer.DumpWordDict2(arabicLex)
        ostream.write(s)
        ostream.close()
def DumpOovOfText(args):
    oovNormalizer =OovNormalizer(args)
    oovNormalizer.LoadDicts()
    ostream_sel = None
    ostream_rmv = None
    if args.tgtdir:
        ostream_sel = open("{0}/text-selected.txt".format(args.tgtdir), 'w', encoding='utf-8')
        ostream_rmv = open("{0}/text-removed.txt".format(args.tgtdir), 'w', encoding = 'utf-8')
    removed_text = str()
    if args.text_file:
        with open(args.text_file, encoding='utf8') as istream:
            for line in istream:
                retList = oovNormalizer.GetLabelAndWords(line, True)
                if not retList: continue
                num_of_oov = oovNormalizer.AddWordsToOovDict(retList[1])
                if num_of_oov >= args.oov_thresh_in_utterance:
                    if args.tgtdir: ostream_rmv.write("{0} {1}\n".format(retList[0], retList[1]))
                elif args.tgtdir: ostream_sel.write("{0} {1}\n".format(retList[0], retList[1]))
        oovWords = oovNormalizer.DumpOovDict()
    if args.tgtdir:
        ostream = open("{0}/oov-count.txt".format(args.tgtdir), 'w', encoding='utf-8')
        ostream.write("{0}".format(oovWords))
        ostream.close()
        ostream_sel.close()
        ostream_rmv.close()
    
"""
  begin functions to do test
"""
def ReadChineseNumberTest(oovNormalizer):
    from random import seed, randint, uniform
    random.seed(777)
    for x in [randint(0,10000) for p in range(100)]:
        verbatim1 = mipinumreader.ReadNormalNumberWithBilingual(str(x), 'chinese')
        verbatim2 = mipinumreader.ReadNormalNumberWithBilingual(str(x), 'english')
        print("number = {0}, verbatim = {1} ({2})".format(x, verbatim1,verbatim2))
        verbatims = mipinumreader.ReadNormalNumberInEnglishMutable(str(x))
        print ("{0}".format(verbatims))
    for x in [uniform(0, 10) for p in range(100)]:
        verbatim1 = mipinumreader.ReadNormalNumberWithBilingual(str(x), 'chinese')
        verbatim2 = mipinumreader.ReadNormalNumberWithBilingual(str(x), 'english')
        print("number = {0}, verbatim = {1} ({2})".format(x, verbatim1, verbatim2))
    logger.info("test ReadNormalNumberInChineseMutable")
    for x in  ['0.15', '3.4']:    # ['210', '120', '12589456', '15', '1400', '1040', '012', '0.15', '3.4']:
        # verbatims = mipinumreader.ReadNormalNumberInChineseMutable(x)
        # print("{0}".format(verbatims))
        # verbatims = mipinumreader.ReadNormalNumberInEnglishMutable(x)
        # print("{0}".format(verbatims))
        xDict = dict()
        oovNormalizer.WriteArabicNumberLexicon(x, xDict)
        for word in xDict:
            for pron in xDict[word]:
                print("{0}\t{1}".format(word, pron))
    for x in ['一千二百', '一千二百四十六', '一千二百三十一', '壹仟贰佰', '一千二', '一千两百零五', '一千〇二']:
        number = mipinumreader.ReadThousandLevelNumber(x)
        print ("chinese={0}, number={1}".format(x, number))
    for x in ['一万四千','一万零四', '两万二', '两万四千八百七十', '五千万']:
        number = mipinumreader.Read10ThousandLevelNumber(x)
        print ("chinese={0}, number={1}".format(x, number))
    for x in ['一亿五千万','一亿零两百万']:
        number = mipinumreader.ReadUpToBillionNumber(x)
        print ("chinese={0}, number={1}".format(x, number))
    for x in ['幺三零九七八七','五十五', '一亿五千万', '柒佰贰拾', '五万三千二百']:
        number = mipinumreader.ReadChineseNumber(x)
        print ("chinese={0:20}, number={1}".format(x, number))
    for utterance in ['新加坡 香格里拉 酒店 均价　是　六百　新币　每晚', '最高价　三千　二百　五十五', '评价　三　点　四　分']:
        newUtterance = mipinumreader.TransferChineseNumberInUtterance(utterance)
        print("utterance = {0}, newUtterance = {1}".format(utterance, newUtterance))
def ChineseSegmentationTest():
    with open("/home2/wjc505/mipitalk_data/mobile-dat-sept07/text", encoding='utf-8') as istream:
        for line in istream:
            line0 = line
            if re.search(r'[\u4e00-\u9fa5]', line):
                line = re.sub(r"([A-Za-z0-9])([\u4e00-\u9fa5])", r'\1 \2', line)
                line = re.sub(r"([\u4e00-\u9fa5])([A-Z]a-z0-9)", r'\1 \2', line)
                print("line={0}    updated_line={1}".format(line0, line))
def TestLabelPronunciationForWordSequence(args):
    oovNormalizer = OovNormalizer(args)
    oovNormalizer.LoadDicts()
    s = str()
    for wordSequence in ['i love games', 'basketball']:
        wordList = wordSequence.split()
        s2sDict = oovNormalizer.LabelPronunciationForWordSequence(wordList)
        logger.info("pronDict={0}".format(s2sDict))
        for word in s2sDict:
            for pron in s2sDict[word]:
                s += "{0}\t{1}\n".format(word, pron)
    print ("{0}".format(s))
def TestConvertEnglishNumberInUtterance(oovNormalizer):
    for utterance in ['the price is three thousand five hundred dollars', 'the price is one hundred point seven eight dollars', 'nineteen eighty']:
        transfered = mipinumreader.TransferEnglishNumberInUtterance(utterance)
        print ("{0}".format(transfered))
"""
    end functions to do test
"""
def TestEntrance(args):
    oovNormalizer = OovNormalizer(args)
    oovNormalizer.LoadDicts()
    if args.test_chinese:
        logger.info("conduct ReadChineseNumberTest")
        ReadChineseNumberTest(oovNormalizer)
    if args.test_english:
        TestConvertEnglishNumberInUtterance(oovNormalizer)
    if args.test_chinese_segmentation:
        ChineseSegmentationTest()
    if args.test_label_pronunciation_for_word_sequence:
        TestLabelPronunciationForWordSequence(args)
    if args.test_read_arabic_number_in_english:
        TestReadArabicNumberInEnglish(args)
def main():
    args = get_args()
    if args.test:
        TestEntrance(args)
    # oovNormalizer = OovNormalizer(args)
    # oovNormalizer.LoadDicts()
    if args.normalize_mipitalk_mobile_data:
        NormalizeMipitalkMobileData(args)
    if args.read_oov_for_transfer_dict:
        oovNormalizer.ReadOovWordList()
    if args.collect_oov_from_transfer_dict:
        oovNormalizer.CollectOovOfTransferDict()
    if args.transfer_text:
        TransferMipitalkMobileData(args)
    if args.dump_oov_of_text:
        DumpOovOfText(args)
    if args.normalize_text:
        NormalizeText(args)
    if args.normalize_text_with_uttid:
        NormalizeTextWithUttId(args)

if __name__ == "__main__":
    main()
