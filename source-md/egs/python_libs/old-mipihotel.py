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
 
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler)

class MipiHotel(object):
    def __init__(self):
        self._mipinumber = mipinumber_lib.MipiNumber()
        self._dict = dict()
        self._checkDict = dict()
        self._history = None
        # self._query = dict()
        self._star = None
        self._queryKeywordDict = dict()
        self._utterance = None
        self._uttList = list()
        self._numberList = list()
        self._floatList = list()
        self._resultDict = dict()
    def ResetSearch(self):
        self._resultDict.clear()
    def GetDict(self):
        return self._dict
    def GetTheHotel(self, hotelId):
        if 'id' not in self._dict:
            self._dict['id'] = dict()
        theHotel = self._dict['id']
        if hotelId not in theHotel:
            theHotel[hotelId] = dict()
        return theHotel[hotelId]
   
    def InsertQueryDict(self, keyId, key, value):
        hotelDict = self._dict
        if keyId not in hotelDict:
            hotelDict[keyId] = dict()
        queryDict = hotelDict[keyId]
        if key not in queryDict:
            queryDict[key] = dict()
        valueDict = queryDict[key]
        if value not in valueDict:
            valueDict[value] = int(0)
        else:
            valueDict[value] += 1
    def GetHotelIdDict(self, word):
        if word not in self._checkDict:
            return None
        idNameDict = self._checkDict[word]
        hotelIdDict = dict()
        nCount = 0
        hotelDict = self._dict
        for idName in idNameDict:
            # print('idName={0}'.format(idName))
            hotelTableWithIdName = hotelDict[idName]
            hotelIdRecord = hotelTableWithIdName[word]
            # for hotelId in hotelIdRecord:
                # print('word={0}, hotelId={1}'.format(word, hotelId))
            if nCount == 0: hotelIdDict = copy.deepcopy(hotelIdRecord)
            else: hotelIdDict.update(hotelIdRecord)
            nCount += 1
        return hotelIdDict
    def InsertCheckDict(self, key, keyId):
        if key not in self._checkDict:
            self._checkDict[key] = dict()
        checkDict = self._checkDict[key]
        if keyId in checkDict:
            checkDict[keyId] += 1
        else:
            checkDict[keyId] = int(0)
    def LoadTable(self, inputFile):
        """ load the overall table
        """
        fieldIdList = list()
        hotelDict = self._dict
        with open(inputFile, 'r') as istream:
            lineNum = 0
            for line in istream:
                line = line.strip()
                if not line: continue
                if lineNum == 0:
                    fieldIdList = [x for x in line.split() if x.strip()]
                    lineNum += 1
                    continue
                hotelInfo  = [x for x in line.split('\t') if x.strip()]
                if len(hotelInfo) != len(fieldIdList):
                    raise Exception('bad line {0}, hotel has {1} items, fields are {2}'.format(line, len(hotelInfo), len(fieldIdList)))
                hotelId = hotelInfo[0]
                theHotel = self.GetTheHotel(hotelId)
                for idx in range(1, len(hotelInfo)):
                    theHotel[fieldIdList[idx]] = hotelInfo[idx]
                    if fieldIdList[idx] != 'name' and fieldIdList[idx] != 'address':
                         self.InsertQueryDict(fieldIdList[idx], hotelInfo[idx], hotelId)
                         self.InsertCheckDict(hotelInfo[idx], fieldIdList[idx])
                lineNum += 1

    def LoadQueryKeywordTable(self, idName, inputFile):
         hotelDict = self._dict
         with open(inputFile, 'r') as istream:
             for line in istream:
                 if not line.strip(): continue
                 lineList = [x for x in line.split() if x.strip()]
                 hotelId = lineList[0]
                 for idx in range(1, len(lineList)):
                     keyword = lineList[idx]
                     self.InsertQueryDict(idName, keyword, hotelId)
                     self.InsertCheckDict(keyword, idName)
    def DumpTable(self, idName):
        """ dump a table
        """
        hotelDict = self._dict
        if not hotelDict:
            raise Exception('empty dict')
        tableDict = hotelDict[idName]
        if not tableDict:
            raise Exception('no table for id {0}'.format(idName))
        overallTable = str()
        if idName == 'id':
            for hotelId in tableDict:
                theHotel = tableDict[hotelId] 
                overallTable += '{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n'.format(hotelId, theHotel['name'], theHotel['star'], theHotel['rate'], theHotel['price'], theHotel['address'])
        else:
            for keyword in tableDict:
                objectDict = tableDict[keyword]
                for value in objectDict:
                    overallTable += '{0}\t{1}\n'.format(value, keyword)
        print('{0}'.format(overallTable))
    def DumpQuery(self):
        checkDict = self._checkDict
        checkMap = str()
        for query in checkDict:
            mapDict = checkDict[query]
            mapStr = str()
            for name in mapDict:
                mapStr += '{0} '.format(name)
            mapStr = mapStr.strip()
            checkMap += '{0}\t{1}\n'.format(query, mapStr)
        print('{0}\n'.format(checkMap))

        """ understanding part
        """
    def NormalizeAndCopy(self, uttList):
        del self._uttList[:]
        for utterance in uttList:
            utterance = utterance.strip()
            if not utterance: continue
            newUtterance = self._mipinumber.ChineseToArabicNumberTransfer(utterance)
            self._uttList.append(newUtterance)
            
    def HasHotelRequest(self):
        keyword = u'酒店|大酒店|公寓|饭店|标准间|豪华间|双人间|单人间|套间|度假村|hotel|apartment'
        for utterance in self._uttList:
            if re.findall(keyword, utterance):
                self._utterance = utterance
                return True
        return False
    def InsertQueryName(self, word):
        if word in self._queryKeywordDict:
            self._queryKeywordDict[word] += 1
        else:
            self._queryKeywordDict[word] = int(0)
    def FillResultDict(self, word):
        """ fill search result in into the self._resultDict,
        intersection will be used if necessary.
        """ 
        hotelIdDict = self.GetHotelIdDict(word)
        self.IntersectResultDict(hotelIdDict)
   
    def CheckNameKeyword(self):
        """ this should consider context as well. So far, we only consider
            isolated keyword case.
        """ 
        self._queryKeywordDict.clear()
        for utterance in self._uttList:
            if not utterance: continue
            for word in [x for x in utterance.split() if x.strip()]:
                if re.findall('\d+[\.]\d*', word): continue 
                word = '{0}'.format(word)
                if word in self._checkDict:
                    self.InsertQueryName(word)
                    self.FillResultDict(word)
            if self._queryKeywordDict:
                return True
            return False
    def IntersectResultDict(self, resultDict):
        """ do dictionary intersection between resultDict and self._resultDict
        """
        if not resultDict: return
        if not self._resultDict:
            self._resultDict = copy.deepcopy(resultDict)
        else:
            tmpDict = copy.deepcopy(self._resultDict)
            for hotelId in tmpDict:
                if hotelId not in resultDict:
                    del self._resultDict[hotelId]
    def CheckPriceRequest(self):
        keyword = u'元钱|块钱|价位|价钱|价格|价|元|price|budget|dollar'
        for utterance in self._uttList:
            self._numberList = self._mipinumber.NumberExtractable(utterance, 50, 10000, True)
            if self._numberList:
                break
                # if re.findall(keyword, utterance):  # this will be addressed in the future
                #    break
        resultDict = dict()
        if self._numberList:
            numberList = list()
            if len(self._numberList) == 1:
                number = int(self._numberList[0])
                numberList.append(number-100)
                numberList.append(number+100)
            else:
                numberList = sorted(self._numberList)
            nCount = 0
            for x in xrange(numberList[0], numberList[1], 10):
                strx = str(x)
                hotelIdDict = self.GetHotelIdDict(strx)
                if hotelIdDict:
                    if nCount == 0:
                        resultDict = copy.deepcopy(hotelIdDict)
                    else:
                        resultDict.update(hotelIdDict)
                    nCount += 1
        self.IntersectResultDict(resultDict)
            
    def CheckRateRequest(self):
        for utterance in self._uttList:
            self._floatList = self._mipinumber.NumberExtractable(utterance, 3.0, 5.0, False)
            if self._floatList:
                break
        resultDict = dict()
        if self._floatList:
            floatList = list()
            if len(self._floatList) == 1:
                number = self._floatList[0]
                floatList.append(number - 0.1)
                floatList.append(number + 0.1)
            else:
                floatList = sorted(self._floatList)
            nCount = 0
            for x in self._mipinumber.frange(floatList[0], floatList[1], 0.1):
                strx=str(x)
                hotelIdDict = self.GetHotelIdDict(strx)
                if hotelIdDict:
                    if nCount == 0:
                        resultDict = copy.deepcopy(hotelIdDict)
                    else:
                        resultDict.update(hotelIdDict)
                    nCount += 1
        self.IntersectResultDict(resultDict)

    def IsQueryHotel(self, uttList):
        """ to check if hotel is asked
        """
        self.NormalizeAndCopy(uttList)
        if not self.HasHotelRequest():
            return False
        self.CheckNameKeyword()
        self.CheckPriceRequest()
        self.CheckRateRequest()
        return True
    def DumpResultDict(self):
        retResult = ''
        if self._resultDict:
            tableDict = self._dict['id']
            for hotelId in self._resultDict:
                # print('hotelId={0}'.format(hotelId))
                theHotel = tableDict[hotelId]
                retResult += '{0}\t{1}\t{2}\t{3}\t{4}\n'.format(theHotel['name'], theHotel['star'], theHotel['rate'], theHotel['price'], theHotel['address'])
        else:
            retResult = u'No search result for query({0})!'.format(self._utterance)
        print ('{0}'.format(retResult))
                
    def GetResultStr(self):
        return self.DumpResultDict()
        
