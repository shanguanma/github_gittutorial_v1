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
import python_libs.mipiutterance as mipiutt_lib
import python_libs.mipitable as mipitable_lib
import warnings
import copy
from bs4 import BeautifulSoup
 
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class MipiRestaurant(object):
    def __init__(self):
        self._mipinumber = mipinumber_lib.MipiNumber()
        self._mipitable = mipitable_lib.MipiTable()
        self._mipiutterance = None
    def LoadClueKeywordDict(self, inputFile):
        self._mipitable.LoadDict('clue', inputFile)
    def LoadCheckAndSearchTable(self, tableName, inputFile):
        self._mipitable.LoadCheckAndSearchTable(tableName, inputFile)
    def ResetSearchResult(self):
        self._mipitable.ResetSearchResult()
    def LoadRestaurant(self, restXmlFile):
        with open(restXmlFile, 'r') as istream:
            soup = BeautifulSoup(istream, 'html.parser')
            for rest in soup.find_all('restaurant'):
                restId = rest.find('id').get_text().strip()
                restName = rest.find('name').get_text().strip()
                self._mipitable.BuildObjectTable(restId, 'name', restName)
                restRate = rest.find('rate').get_text().strip()
                restRate = '{0:.1f}'.format(float(restRate))
                self._mipitable.BuildObjectTable(restId, 'rate', restRate)
                self._mipitable.BuildCheckAndSearchTable('rate', restRate, restId)
                restPrice = rest.find('price').get_text().strip()
                priceList = list()
                self._mipinumber.AddRMBIdentifierToPrice(restPrice, priceList)
                # logger.info('restId={3}, restName={0}, restPrice={1}, priceList={2}'.format(restName, restPrice, priceList[0], restId))
                self._mipitable.BuildObjectTable(restId, 'price', priceList[0])
                self._mipitable.BuildCheckAndSearchTable('price', str(priceList[1]), restId)
                restPhone = rest.find('phone').get_text().strip()
                self._mipitable.BuildObjectTable(restId, 'phone', restPhone)
                restAddress = rest.find('address').get_text().strip()
                self._mipitable.BuildObjectTable(restId, 'address', restAddress)
                restCuisine = rest.find('cuisine').get_text().strip()
                self._mipitable.BuildObjectTable(restId, 'cuisine', restCuisine)

    def _HasRestaurantRequestRule1(self):
        keyword = u'餐馆|餐厅|食阁|restaurant|food court|吃饭|have lunch|have dinner'
        return self._mipiutterance.SearchKeyword(keyword)
    def _HasRestaurantRequestRule2(self):
        keyword = u'茶馆|咖啡店|星巴克|starbucks|早点|下午茶|宵夜|tea break|coffe break'
        return self._mipiutterance.SearchKeyword(keyword)
    def _HasRestaurantRequestRule3(self):
        keyword = u'酒馆|wine bar|酒吧|啤酒|bear'
        return self._mipiutterance.SearchKeyword(keyword)
    def _HasRestaurantRequestRule4(self):
        return self._mipiutterance.IsIntersected(self._mipitable.GetTable('clue', False))
      
    def HasRestaurantRequest(self):
        ''' use thumb of rule to check if restaurant query is requested '''
        if self._HasRestaurantRequestRule1():
            return True
            logger.info('confirmed from rule1')
            return True
        if self._HasRestaurantRequestRule2():
            logger.info('confirmed from rule2')
            return True
        if self._HasRestaurantRequestRule3():
            logger.info('confirmed from rule3')
            return True
        if self._HasRestaurantRequestRule4():
            logger.info('confirmed from rule4')
            return True
        logger.info('no restaurant query got checked')
        return False
            
    def IsQueryRestaurant(self, mipiUtterance):
        ''' check if restaurant query occurrs. '''
        self._mipiutterance = mipiUtterance
        if not self.HasRestaurantRequest():
            return False
        self._mipitable.SetUtterance(self._mipiutterance)
        nameTypeDict = {'name':0, 'address':0, 'cuisine':0}
        self._mipitable.CheckNameKeyword(nameTypeDict)
        self._mipitable.CheckPriceRequest()
        self._mipitable.CheckRateRequest()
        return True
    def DumpTable(self, tableName, fromBigTable=False):
        return self._mipitable.DumpTable(tableName, fromBigTable)
    def DumpResult(self):
        retResult = ''
        resultDict = self._mipitable.GetResultTable()
        if resultDict:
            tableDict = self._mipitable.GetTable('id', False)
            for restId in resultDict:
                theRest = tableDict[restId]
                retResult += '{0}\n{1}\n{2}\n{3}\n{4}\n\n'.format(theRest['name'], theRest['rate'], theRest['price'], theRest['phone'], theRest['address'], theRest['cuisine'])
        else:
            retResult = u'No search results for your query ({0})'.format(self._mipiutterance.GetUtterance())
        return retResult
""" the end of MipiRestaurant class """
