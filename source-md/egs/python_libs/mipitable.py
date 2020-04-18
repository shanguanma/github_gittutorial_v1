# 2017 Haihua Xu 
# -*- coding: utf-8 -*-
from __future__ import print_function
import re
import os
import warnings
import sys
reload(sys)
import copy
sys.setdefaultencoding('utf8')
sys.path.insert(0, 'source/egs')
import python_libs.mipinumber as mipinumber_lib
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class MipiTable(object):
    def __init__(self):
        self._dict = dict()
        self._checkDict = dict()
        self._resultDict = dict()
        self._queryKeywordDict = dict()
        self._mipinumber = mipinumber_lib.MipiNumber()
        self._mipiutterance = None
    def DumpQueryKeyword(self):
        query = str()
        for word in self._queryKeywordDict:
            query += word + ' '
        return query.strip()
    def SetUtterance(self, mipiutterance):
        self._mipiutterance = mipiutterance
    def ResetSearchResult(self):
        self._resultDict.clear()
    def GetResultTable(self):
        return self._resultDict
    def GetCheckTable(self):
        return self._checkDict
    def GetTable(self, tableName=None, insert=True):
        if not tableName:
            return self._dict
        return self.GetTableDict(tableName, insert)
    def GetTableDict(self, searchType, insert=True):
        if searchType not in self._dict:
            if insert:
                self._dict[searchType] = dict()
            else:
                # raise Exception('unexpected table Id {0}'.format(searchType))
                return dict()
        return self._dict[searchType]
    def GetTheObjectTable(self, objId):
        theObjectIdTable = self.GetTableDict('id')
        if objId not in theObjectIdTable:
            theObjectIdTable [objId] = dict()
        return theObjectIdTable[objId]
    def BuildObjectTable(self, objId, idKey, idValue):
        theObject = self.GetTheObjectTable(objId)
        theObject[idKey] = idValue
    def BuildCheckAndSearchTable(self, searchType, searchKey, searchValue):
        """ Frist build check table, then build search table
        """
        searchDict = self.GetTableDict(searchType)
        checkDict = self._checkDict
        if searchKey not in checkDict:
            checkDict[searchKey] = dict()
        dictOfDict = checkDict[searchKey]
        if searchType in dictOfDict:
            dictOfDict[searchType] += 1
        else:
            dictOfDict[searchType] = int(0)
        """Now fill search dict"""
        if searchKey not in searchDict:
            searchDict[searchKey] = dict()
        valueDict = searchDict[searchKey]
        if searchValue not in valueDict:
            valueDict[searchValue] = int(0)
        else:
            valueDict[searchValue] += 1
    def LoadDict(self, searchType, inputFile):
        tgtDict = self.GetTableDict(searchType)
        with open(inputFile, 'r') as istream:
            for line in istream:
                line = line.strip()
                if not line: continue
                m = re.search(r'^(\S+)\s*(.*)$', line)
                if m:
                    key = m.group(1)
                    value = m.group(2)
                    tgtDict[key] = value
    def LoadCheckAndSearchTable(self, searchType, inputFile):
        """ Load check and search table from file
            the file format should be like typeIndexId   searchKey1 searchKey2 ... searchKeyN
        """
        with open(inputFile, 'r') as istream:
            for line in istream:
                line = line.strip()
                valueKeyLine = [ x for x in line.split() if x.strip()]
                if len(valueKeyLine) <= 1: continue
                searchValue = valueKeyLine[0]
                valueKeyLine.pop(0)
                for searchKey in valueKeyLine:
                    serachKey = searchKey.strip()
                    self.BuildCheckAndSearchTable(searchType, searchKey, searchValue)
    def DumpTable2(self, tableDict, keyFirst=True):
        """ dump dict of dict """
        tableStr = ''
        for searchKey in tableDict:
            dictOfDict = tableDict[searchKey]
            for searchValue in dictOfDict:
                line = str()
                if keyFirst:
                    line = '{0}\t{1}\n'.format(searchKey, searchValue)
                else:
                    line = '{0}\t{1}\n'.format(searchValue, searchKey)
                tableStr += line
        return tableStr
    def DumpTable(self, searchType, fromBigTable=False):
        tableStr = str()
        if fromBigTable:
            objectIdTable = self.GetTableDict('id', False)
            for restId in objectIdTable:
                theObjectTable = self.GetTheObjectTable(restId)
                if searchType not in theObjectTable:
                    raise Exception('no search type {0}'.format(searchType))
                typeInfo = theObjectTable[searchType]
                line='{0}\t{1}\n'.format(restId, typeInfo)
                tableStr += line
        else:
            tableDict = self.GetTableDict(searchType, False)
            tableStr = self.DumpTable2(tableDict)
        return tableStr
    def DumpCheckTable(self, tableId):
        tableStr = str()
        for keyword in self._checkDict:
            searchDict = self._checkDict[keyword]
            for xId in searchDict:
                if xId == tableId:
                    line = '{0}\t{1}\n'.format(keyword, xId)
                    tableStr += line
        return tableStr
    def RemoveNumericWord(self, tgtDict):
        for word in list(tgtDict):
            if re.findall('^\d+([\.]\d*)?$', word):
                del tgtDict[word]
    def DumpDict2(self, dict1):
        for keyword in dict1:
            dict2 = dict1[keyword]
            for keyword2 in dict2:
                print('{0}\t{1}\n'.format(keyword, keyword2))
    def DictIntersection(self, dict1, dict2, nameTypeDict):
        intersectDict = dict()
        for word in dict1:
            if word in dict2:
                typeDict = dict2[word]
                for queryType in typeDict:
                    if queryType in nameTypeDict:
                        intersectDict[word] = dict2[word]
        return intersectDict
    def ObjectIdDictIntersection(self, srcDict, tgtDict):
        ''' Get Intersections of srcDict and tgtDict, if 
            tgtDict is empty, we copy srcDict to tgtDict, otherwise, we intersect them
            and put result into the tgtDict '''
        if not srcDict: return
        if not tgtDict:
            tgtDict = copy.deepcopy(srcDict)
        else:
            tmpDict = copy.deepcopy(tgtDict)
            for objId in tmpDict:
                if objId not in srcDict:
                    del tgtDict[objId]
        return tgtDict
    def GetObjectIdDict(self, word):
        """ use word as keyword, supDict as supervising dict to search in tgtDict """
        supDict = self._checkDict
        tgtDict = self._dict
        if word not in supDict:
            return None
        idNameDict = supDict[word]
        retIdDict = dict()
        for idName in idNameDict:
            tgtTable = tgtDict[idName]
            objIdDict = tgtTable[word]
            retIdDict.update(objIdDict)
        return retIdDict
    """Search part"""
    def FillResultDict(self, word):
        objectIdTable = self.GetObjectIdDict(word)
        # logger.info('{0}'.format(restIdDict))
        if not objectIdTable: return
        self._resultDict =copy.deepcopy( self.ObjectIdDictIntersection(objectIdTable, self._resultDict))
        if not self._resultDict:
            warnings.warn("Sorry, no intersect results for keyword '{0}'".format(self.DumpQueryKeyword()))
    def CheckNameKeyword(self, nameTypeDict):
        """ keyword based  search """
        self._queryKeywordDict = self.DictIntersection(self._mipiutterance.GetWordDict(), self._checkDict, nameTypeDict)
        if not self._queryKeywordDict:
            return False
        for word in self._queryKeywordDict:
            self.FillResultDict(word)
        return True

    def CheckPriceRequest(self):
        numList = self._mipiutterance.GetIntegerList(True)
        if not numList:
            return
        logger.info('numList for price={0}'.format(numList))
        # print('{0}'.format(self.DumpCheckTable('price')))
        numList = self._mipinumber.GetApproxNumList(numList, 10, 10000, 10, True)
        restIdDict = dict()
        lower = numList[0]
        upper = numList[-1]
        # logger.info('lower={0}, upper={1}'.format(lower, upper))
        for priceNum in xrange(lower, upper, 10):
            word = str(priceNum)
            if word in self._checkDict:
                idDict = self._checkDict[word]
                for idName in idDict:
                    if idName == 'price':
                        restIdDict.update(self.GetObjectIdDict(word))
        # logger.info('restIdDict={0}'.format(restIdDict))
        if numList:
            if not restIdDict:
                warnings.warn('Sorry, no restaurant searched with your price range [{0}, {1}].'.format(numList[0], numList[len(numList)-1]))
            else:
                logger.info('queryKeyword={0}'.format(self.DumpQueryKeyword()))
                # logger.info('resultDict={0}'.format(self._resultDict))
                self.ObjectIdDictIntersection(restIdDict, self._resultDict)
                self.CheckObjectIdByPrice(lower, upper)
    def CheckObjectIdByPrice(self, lowerbound, upperbound):
        objectTable  = self.GetTable('id', False)
        for objId in list(self._resultDict):
            theObject = objectTable[objId]
            priceNum = self._mipinumber.GetRawPrice(theObject['price'])
            if priceNum < lowerbound or priceNum >= upperbound:
                del self._resultDict[objId]
    def CheckRateRequest(self):
        numList = self._mipiutterance.GetRealNumList(1.0, 5.0)
        logger.info('numList={0}'.format(numList))
        numList = self._mipinumber.GetApproxNumList(numList, 0.0, 5.0, 0.1)
        logger.info('After approx. numList={0}'.format(numList))
        restIdDict = dict()
        for rateNum in numList:
            word = '{:.1f}'.format(rateNum)
            logger.info('rateNum={0:.2f}, word={1}'.format(rateNum, word))
            if word in self._checkDict:
                idDict = self._checkDict[word]
                for idName in idDict:
                    if idName == 'rate':
                        restIdDict.update(self.GetObjectIdDict(word))
        if numList:
            if not restIdDict:
                logger.info('Sorry, no restaurant searched with rate range [{0}, {1}]'.format(numList[0], numList[len(numList)-1]))
            else:
                self.ObjectIdDictIntersection(restIdDict, self._resultDict) 
