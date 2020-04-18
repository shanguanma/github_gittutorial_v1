# 2017 Haihua Xu 
# -*- coding: utf-8 -*-
import re
import os
import copy
import sys
reload(sys)
sys.setdefaultencoding('utf8')
sys.path.insert(0, 'source/egs')
import python_libs.mipinumber as mipinumber_lib
import mipitable
from bs4 import BeautifulSoup
 
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class MipiUtterance(object):
    def __init__(self):
        self._mipinumber = mipinumber_lib.MipiNumber()
        self._uttList = list()
        self._utterances = str()
        self._hitutterance = str()
        self._wordDict = dict()

    def NormalizeAndCopy(self, uttList):
        del self._uttList[:]
        self._utterances = str()
        for utterance in uttList:
            utterance = utterance.strip()
            if not utterance:continue
            # logger.info('utterance before transfer: {0}'.format(utterance))
            newUtterance = self._mipinumber.ChineseToArabicNumberTransfer(utterance)
            # logger.info('utterance after transfer: {0}'.format(newUtterance))
            self.InitWordDictWithUtterance(newUtterance)
            self._uttList.append(newUtterance)
            self._utterances += newUtterance + '\n'
    def InitWordDictWithUtterance(self, utterance):
        wordList = [x for x in utterance.strip().split() if x.strip()]
        self._wordDict.clear()
        for word in wordList:
            word = '{0}'.format(word.strip())
            if word in self._wordDict:
                self._wordDict[word] += 1
            else:
                self._wordDict[word] = int(1)
    def GetUtterance(self):
        return self._utterances
    def InitUttList(self, uttList):
        self.NormalizeAndCopy(uttList)
    def GetWordDict(self):
        return self._wordDict
    def SearchKeyword(self, keyword, ):
        for utterance in self._uttList:
            if re.findall(keyword, utterance):
                self._hitutterance = utterance
                return True   
    def GetHitUtterance(self):
        return self._hitutterance
    def IsIntersected(self, srcDict):
        if not srcDict: return False
        for word in self._wordDict:
            if word in srcDict:
                return True
        return False

    def GetIntegerList(self, positive=False):
        intList = list()
        for word in self._wordDict:
            if self._mipinumber.IsInteger(word):
                if positive:
                    if int(word) > 0:
                        intList.append(int(word))
                else:
                    intList.append(int(word))
        return intList
    def GetRealNumList(self, lowerBound, upperBound):
        ''' Get those numbers between [lowerBound, upperBound]. '''
        realList = list()
        for word in self._wordDict:
            if self._mipinumber.IsNoSignNumber(word):
                try:
                    realNum = float(word)
                    if realNum >= lowerBound and realNum <= upperBound:
                        realList.append(realNum)
                except ValueError: continue
        return realList
                    
                
