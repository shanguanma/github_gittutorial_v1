# 2017 Haihua Xu 
# -*- coding: utf-8 -*-
import re
import os

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler)

""" define a class to deal with dict
"""
class MipiDictVege(object):
    def __init__(self, lowerWord=True):
        self._wordPronDict = dict()
        self._wordCountDict = dict()
        self._phoneCountDict = dict()
        self._lowercase = lowerWord

    def GetWordPronDict(self):
        return self._wordPronDict

    def GetPronDict(self, word):
        return self._wordPronDict[word]

    def CountPhone(self, phoneList, phoneDict):
        for phone in phoneList:
            if phone in phoneDict:
                phoneDict[phone] += 1
            else:
                phoneDict[phone] = int(1)
    def CountWord(self, text):
        wordList = [w for w in text.split() if w.strip()]
        self.CountPhone(wordList, self._wordCountDict)
    def CountWord2(self, text, countDict):
        wordList = [w for w in text.split() if w.strip()]
        self.CountPhone(wordList, countDict)
    def DumpCountDict(self, countDict):
        if not countDict:
            return
        import operator
        # print('countDict={0}'.format(countDict))
        sortedList = sorted(countDict, key=countDict.get, reverse=True)
        return sortedList 
    def DumpWordList(self):
        return self.DumpCountDict(self._wordCountDict)
    def AddWordPronToDict(self, word, pron, wordPronDict, phoneDict=None):
        if word in wordPronDict:
            pronDict = wordPronDict[word]
            if pron in pronDict:
                pronDict[pron] += 1
            else:
                pronDict[pron] = int(0)
        else:
            wordPronDict[word] = dict()
            pronDict = wordPronDict[word]
            pronDict[pron] = int(0)
        if phoneDict:
            phoneList = [p for p in pron.split() if p.strip()]
            if len(phoneList) <= 0:
                raise Exception("empty pronunciation for word {0}".format(word))
                self.CountPhone(phoneList, phoneDict)

    def LoadDict(self, dictFile, wordPronDict=None, phoneDict=None):
        if not dictFile:
            logger.debug('dict file expected')
            return
        with open(dictFile, 'r') as iStream:
            for line in iStream:
                if not line.strip():continue
                m = re.search(r'(^\S+)(.*)$', line.strip())
                word = m.group(1)
                pron = m.group(2)
                pron = pron.strip()
                if not (word and pron): continue
                if self._lowercase:
                    word = word.lower()
                targetWordPronDict = self._wordPronDict
                phoneCountDict = self._phoneCountDict
                if wordPronDict:
                    targetWordPronDict = wordPronDict
                    phoneCountDict = phoneDict
                self.AddWordPronToDict(word, pron, targetWordPronDict, phoneCountDict)
    def LoadWordList(self, wordListFile, wordListDict):
        with open(wordListFile, 'r') as istream:
            for line in istream:
                wordList = [x for x in line.split() if x.strip()]
                if wordList:
                    self.CountPhone(wordList, wordListDict)
    def DumpPhoneList(self):
        self.DumpCountDict(self._phoneCountDict)
                    
""" end of MipiDictVege
"""
class MipiWorldDict(object):
    def __init__(self):
       self._dict = dict()
       self._cityDict = dict()
       self._hotelDict = dict()
       self._restuarantDict = dict()
       self._mrtDict = dict()
       self._city = None
       self._hotelQuery = dict()
       self._restaurantQuery = dict()
       self._mrtQuery = dict()

    def GetCityDict(self, city):
        if not city:
            raise Exception('city should not be empty')
        self._city = city
        if city in self._dict:
            self._cityDict = self._dict[city]
            return self._dict[city]
        self._dict[city] = dict()
        self._cityDict = self._dict[city]
        return self._cityDict

    def GetCategoryDict(self, category, city=None):
        city = self._city
        if not city:
            raise Exception('city is not specified')
        cityDict = self.GetCityDict(city)
        if category not in cityDict:
            cityDict[category] = dict()
        categoryDict = cityDict[category]
        return categoryDict
    def GetOneHotel(self, hotelId):
        hotelDict = self.GetCategoryDict('hotel')
        if 'id' not in hotelDict:
            hotelDict['id'] = dict()
        oneHotel = hotelDict['id']
        if hotelId in oneHotel:
            return oneHotel[hotelId]
        oneHotel[hotelId] = dict()
        return oneHotel[hotelId]
    def InsertQueryDict(self, keyId, key, value):
        hotelDict = self.GetCategoryDict('hotel')
        if keyId not in hotelDict:
            hotelDict[keyId] = dict()
        queryDict = hotelDict[keyId]
        if key not in queryDict:
            queryDict[key] = dict()
        objectDict = queryDict[key]
        if value not in objectDict:
            objectDict[value] = int(0)
        else:
            objectDict[value] += 1
    def LoadHotelTable(self, inputFile):
        """ load the overall hotel table
        """
        fieldIdList = list()
        hotelDict = self.GetCategoryDict('hotel')
        with open(inputFile, 'r') as istream:
            lineNum = 0
            for line in istream:
                if not line.strip(): continue
                line = line.strip()
                if lineNum == 0:
                    fieldIdList = [x for x in line.split() if x.strip()]
                    lineNum += 1
                    continue
                hotelInfo  = [x for x in line.split('\t') if x.strip()]
                if len(hotelInfo) != len(fieldIdList):
                    raise Exception('bad line {0}, hotel has {1} items, fields are {2}'.format(line, len(hotelInfo), len(fieldIdList)))
                hotelId = hotelInfo[0]
                theHotel = self.GetOneHotel(hotelId)
                for idx in range(1, len(hotelInfo)):
                    theHotel[fieldIdList[idx]] = hotelInfo[idx]
                    if fieldIdList[idx] != 'name' and fieldIdList[idx] != 'address':
                        self.InsertQueryDict(fieldIdList[idx], hotelInfo[idx], hotelId)
                lineNum += 1
    def LoadNameKeywordTable(self, idName, inputFile):
        """ load hotel name as keyword table
        """
        hotelDict = self.GetCategoryDict('hotel')
        with open(inputFile, 'r') as istream:
            for line in istream:
                if not line.strip(): continue
                lineList = [x for x in line.split() if x.strip()]
                hotelId = lineList[0]
                for idx in range(1, len(lineList)):
                    keyword = lineList[idx]
                    self.InsertQueryDict(idName, keyword, hotelId)
    def DumpTable(self, category, idName):
        """ dump a table, for instance, city=singapore, category=hotel, idname = 'id'
        """
        categoryDict = self.GetCategoryDict(category)
        if not categoryDict:
            raise Exception('dict for category {0} is empty'.format(category))
        tableDict = categoryDict[idName]
        if not tableDict:
            raise Exception('table for id {0} is empty'.format(idName))
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
        print ('{0}'.format(overallTable))
