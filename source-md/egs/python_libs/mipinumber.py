# 2017 Haihua Xu 
# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import os
import warnings
import sys
reload(sys)
sys.setdefaultencoding('utf8')
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler)

""" a class to deal with number 
"""
class MipiNumber(object):
    def __init__(self, unitDict=None, unitLabelDict=None, unitDictApprox=None):
        self._number = None
        self._unit = None
        self._unitDict = unitDict
        self._unitLabelDict = unitLabelDict
        self._unitDictApprox = unitDictApprox
        self._cn2arabicNumberMap = {u'零':0, u'一':1, u'幺':1, u'壹':1, u'二':2,u'两':2, u'贰':2, u'三':3, u'四':4, u'肆':4, u'五':5, u'伍':5, u'六':6, u'陆':6, u'七':7, u'柒':7, u'八':8, u'捌':8, u'九':9, u'玖':9, u'十':10, u'拾':10}
    def IsInteger(self, strToken):
        try: 
            int(strToken)
        except ValueError:
            return False
        return True
    def IsNoSignNumber(self, strToken):
        m = re.search('^(\d+(\.\d+)?)$', strToken)
        if m:
            return True
        return False
    def IsTransferable(self, strToken):
        """ transfer a integer number plus a unit, for instance, 567dollars
        """
        strToken = strToken.strip()
        if re.search('\s', strToken):
            warnings.warn("string {0} is contains more than one word".format(strToken))
            return False
        if not strToken:
            return False
        m = re.search('^(\d+(\.\d+)?)(\S+)$', strToken)
        if m:
            if not self._unitDict:
                warnings.warn('unit dict is not specified for number {0}'.format(strToken))
                return False
            number = m.group(1)
            unit = m.group(3)
            if unit in self._unitDict.GetWordPronDict():
                self._number = number
                self._unit = unit
                return True
            else:
                return False
        return False
    def IsTransferable2(self, word, nextWord):
        if word in self._unitDict.GetWordPronDict() and self.IsNoSignNumber(nextWord):
            self._number = nextWord
            self._unit = word
            return True
        if self.IsNoSignNumber(word) and nextWord in self._unitDict.GetWordPronDict():
            self._number = word
            self._unit = nextWord
            return True
        return False
    def IsApprox(self):
        if not self._unitDictApprox:
            return False
        if self._unit in self._unitDictApprox.GetWordPronDict():
            return True
        return False

    def GetApprox(self):
        if not (self._number and self._unit):
            raise Exception('number or unit is not specified')
        number = self._number
        if self.IsApprox():
            number = self.GetApproxNumber(number)
        return number

    def GetTransfer(self):
        number = self.GetApprox()
        transferToken = "{0}{1}".format(number, self._unit)
        return transferToken

    def LabelNumberAndUnit(self):
        number = self.GetApprox()
        nLabel = self.GetNumberLabel()
        unitLabel = self.GetUnitLabel()
        strLabel = "{0}\t{1}\n".format(number, nLabel)
        strLabel += "{0}\t{1}\n".format(self._unit, unitLabel)
        return strLabel
        
    def GetUnit(self):
        return self._unit

    def GetNumber(self):
        return self._number

    def GetApproxNumber(self, number):
        number = int(float(number))
        if number < 1000:
            quotient = int(number /10)
            remainder = number %10
            if remainder >= 5:
                remainder = 10
            else:
                remainder = 0
            number = 10*quotient + remainder
        else:
            quotient = int(number/100)
            remainder = number %100
            if remainder >=50:
                remainder = 100
            else:
                remainder = 0
            number = 100*quotient + remainder
        return number

    def GetUnitLabel(self):
        if not self._unit:
            raise Exception('unit is not specified')
        if self._unit not in self._unitDict.GetWordPronDict():
            raise Exception('unknown unit {0} or unit dict is not specified'.format(self._unit))
        pronDict = self._unitDict.GetPronDict(self._unit)
        for pron in pronDict:
            return pron

    def GetNumberLabel(self):
        if not self._number:
            raise Exception('number is not specified')
        unitLabel = self.GetUnitLabel()
        # print('unitLabel={0}'.format(unitLabel))
        if not self._unitLabelDict:
            raise Exception('unit label dict not specified')
        if unitLabel not in self._unitLabelDict.GetWordPronDict():
            raise Exception('unindentified unitlabel {0} in unit label dict'.format(unitLabel))
        pronDict = self._unitLabelDict.GetPronDict(unitLabel)
        allProns = ''
        for pron in pronDict:
            allProns += pron + ' '
        allProns = allProns.strip()
        return allProns

    def IsChineseNumber(self, word):
        keyword = ur'^[零|一|幺|壹|二|两|贰|三|叁|四|肆|五|伍|六|陆|七|柒|八|捌|九|玖|十|拾|百|佰|千|仟|万|萬|亿]+$'
        if re.search(keyword, word):
            return True
        return False
    def NumberConvertible(self, word):
        retList = list()
        self._NumberConvertible(word, 0, retList)
        if not retList:
            return word
        return retList[0]
    def GetMagnitude(self, number):
        n = 0
        while(int(number/10) > 0):
            n += 1
            number = int(number/10)
        return n
    def _NumberConvertible(self, word, accumulated, retList):
        """ maximum number is 999,999,999
        """
        # keyword =ur'^[零|一|幺|壹|二|贰|两|三|叁|四|肆|五|伍|六|陆|七|柒|八|捌|九|玖|十|拾]+$'
        keyword =ur'^[零|一|幺|壹|二|贰|两|三|叁|四|肆|五|伍|六|陆|七|柒|八|捌|九|玖]+$'
        keywordDict = {u'零':0, u'一':1, u'幺':1, u'壹':1, u'二':2,u'两':2, u'贰':2, u'三':3, u'四':4, u'肆':4, u'五':5, u'伍':5, u'六':6, u'陆':6, u'七':7, u'柒':7, u'八':8, u'捌':8, u'九':9, u'玖':9, u'十':10, u'拾':10}
        unitDict = {u'亿':100000000, u'千万':10000000, u'百万':1000000, u'十万':100000, u'万':10000, u'千':1000, u'百':100, u'十':10}
        # print('word-before-search={0}'.format(word), file=sys.stderr)
        if re.search(keyword, word):
            # print('expected here word={0}'.format(word), file=sys.stderr)
            wordList = list(word)
            nLen = len(wordList)
            accuNumber = int(accumulated)
            conStr = ''
            # print ('accumulated={0}'.format(accumulated))
            for idx in range(nLen):
                if wordList[idx] not in keywordDict:
                    raise Exception('oov word {0}'.format(wordList[idx]))
                if accumulated > 0:
                    accuNumber += int(keywordDict[wordList[idx]])
                else:
                    conStr += str(keywordDict[wordList[idx]])
            retValue = conStr
            if accumulated > 0:
                retValue = str(accuNumber)
            del retList[:]
            retList.append(retValue)
            return

        pattern=ur'^(\S)(亿|万|千|百|十)'
        m = re.search(pattern, word)
        if m:
            cnFactor = m.group(1)
            unit = m.group(2)
            if unit not in unitDict:
                raise Exception('unexpected unit {0}'.format(unit))
            factor = keywordDict[cnFactor]
            magnitude = unitDict[unit]
            if magnitude == 10000:
                accumulated += int(factor)
                accumulated *= 10000
            else:
                accumulated += int(factor)* int(magnitude)
            newWord = re.sub(pattern, '', word)
            nextMagnitude = int(magnitude) / 10
            if nextMagnitude >= 10:
                # print('nexMagnitude={0}, newWord={1}'.format(nextMagnitude, newWord))
                newWord = self._ConvertDigitWithoutUnit(nextMagnitude, newWord)
            # print('fator={0}, magnitude={1}, accumulated={2}, residual={3}'.format(factor, magnitude, accumulated, newWord), file=sys.stderr)
            if newWord:
                newWord = re.sub(ur'^零', '', newWord)
                self._NumberConvertible(newWord, accumulated, retList)
            else:
                del retList[:]
                retList.append(str(accumulated))
        else:
            pattern=ur'^(十|万)'
            m = re.search(pattern, word)
            if m:
                nextMagnitude = 0
                unit = m.group(0)
                if unit == u'十':
                    accumulated += int(10)
                elif unit == u'万':
                    accumulated *= 10000
                    nextMagnitude = 1000
                pattern = ur'{0}'.format(unit)
                newWord = re.sub(pattern, '', word)
                if nextMagnitude > 0:
                    newWord = self._ConvertDigitWithoutUnit(nextMagnitude, newWord)
                # print('newWord={0}'.format(newWord))
                if not newWord:
                    del retList[:]
                    retList.append(str(accumulated))
                else:
                    newWord = re.sub(ur'^零','', newWord)
                    self._NumberConvertible(newWord, accumulated, retList)
            else:
                del retList[:]
    def _ConvertDigitWithoutUnit(self, magnitude, word):
        if magnitude < 10:
            return word
        keyword =ur'^[一|壹|二|贰|两|三|叁|四|肆|五|伍|六|陆|七|柒|八|捌|九|玖]$'
        arabic2cnDict={1000:u'千', 100:u'百', 10:u'十'}
        if re.search(keyword, word):
            if magnitude not in arabic2cnDict:
                raise Exception('unexpected magnitude {0}'.format(magnitude))
            unit = arabic2cnDict[magnitude]
            return u'{0}{1}'.format(word, unit)
        return word
    def ArabicChineseReading(self, number):
        pass
    def IsFromSingleNumberByMagnitude(self, numList):
        magList = list()
        idx = 0
        nLen = len(numList)
        if nLen <= 1:
            return False
        for num in numList:
            magList.append(self.GetMagnitude(num))
            idx +=1
            if idx -2 >= 0:
                if(magList[idx-1] >= magList[idx-2]):
                    return False
        return True
            
    def ConcateNumberSequenceIntoSingleNumber(self, utterance):
        findList = re.findall(ur'(?:\d+\s+)+', utterance)
        theLast = re.findall(ur'(?:\d+\s+)+\d+$', utterance)
        if theLast: findList.append(theLast[0])
        for sequence in findList:
            numberList = list()
            for strNumber in [ x for x in sequence.split() if x.strip()]:
                numberList.append(int(strNumber))
            if self.IsFromSingleNumberByMagnitude(numberList):
                newNumber = sum(numberList)
                fromPattern = ur'{0}'.format(sequence)
                toPattern = '{0} '.format(newNumber)
                utterance = re.sub(fromPattern, toPattern, utterance)
        return utterance
    def ChineseToArabicNumberTransfer(self, utterance):
        """ transfer all non-arabic number (both English and Mandain) into
            arabic number
        """ 
        wordList = [ x for x in utterance.split() if x.strip() ]
        newUtterance = ''
        for word in wordList:
            if self.IsChineseNumber(word):
                word = self.NumberConvertible(word)
            newUtterance += word + ' '
        newUtterance = newUtterance.strip()
        newUtterance = self.ConcateNumberSequenceIntoSingleNumber(newUtterance)
        return newUtterance

    def frange(self, start, stop, step):
        x = start
        while x < stop:
            yield x
            x += step
    def NumberExtractable(self, utterance, lowerBound=0, upperBound=10000, approx=False):
        numList = re.findall('\d+[\.]?\d*', utterance)
        if not numList:
            return None
        retList = list()
        for number in numList:
            number = float(number)
            if lowerBound <= number and  number <= upperBound:
                if approx:
                    number = self.GetApproxNumber(number)
                retList.append(number)
        return retList
    def AddRMBIdentifierToPrice(self, strLine, retPriceList):
        del retPriceList[:]
        numList = re.findall(ur'\d+[\.]?\d*', strLine)
        if not numList:
            raise Exception('no number contained in {0}'.format(strLine))
        normalizedPrice = '{0}{1}'.format('¥', numList[0])
        retPriceList.append(normalizedPrice)
        approxPrice = self.GetApproxNumber(numList[0])
        retPriceList.append(approxPrice)
    def GetRawPrice(self, priceStr):
        if not priceStr:
            return int(0)
        priceStr = re.sub('¥', '', priceStr)
        priceStr = re.sub('\.\d+$', '', priceStr)
        if not priceStr:
            return int(0)
        try:
            return int(priceStr)
        except ValueError:
            raise Exception('not a integer ({0})'.format(priceStr))

    def GetApproxNumList(self, numList, lowerBound, upperBound, step, integer=False):
        retNumList = list()
        if not numList:
            return retNumList
        nSize = len(numList)
        if nSize == 1:
            x = numList[0] - step
            if x < lowerBound: x = lowerBound
            lowerBound = x
            x = numList[0] + 1.5*step
            if x > upperBound: x = upperBound
            upperBound = x
        else:
            newNumList = sorted(numList)
            x = numList[0]
            if x < lowerBound: x = lowerBound
            lowerBound = x
            x = numList[1] + step
            if x > upperBound: x = upperBound
            upperBound = x
        for x in self.frange(lowerBound, upperBound, step):
            if integer:
                retNumList.append(int(x))
            else:
                retNumList.append(x)
        return retNumList
""" end of MipiNumber class
"""
