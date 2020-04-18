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

class MipiHotel(object):
    def __init__(self):
        self._mipinumber = mipinumber_lib.MipiNumber()
        self._mipitable = mipitable_lib.MipiTable()
        self._miputterance = None
    
    def LoadClueKeywordDict(self, inputFile):
        self._mipitable.LoadDict('clue', inputFile)
    def LoadCheckAndSearchTable(self, tableName, inputFile):
        self._mipitable.LoadCheckAndSearchTable(tableName, inputFile)
    def ResetSearchResult(self):
        self._mipitable.ResetSearchResult()
    def LoadHotel(self, hotelXmlFile):
        with open(hotelXmlFile, 'r') as istream:
            soup =BeautifulSoup(istream, 'html.parser')
            for hotel in soup.find_all('hotel'):
                hotelId = hotel.find('id').get_text().strip()
                hotelName = hotel.find('name').get_text().strip()
                self._mipitable.BuildObjectTable(hotelId, 'name', hotelName)
                hotelRate = hotel.find('rate').get_text().strip()
                hotelRate = '{0:.1f}'.format(float(hotelRate))
                self._mipitable.BuildObjectTable(hotelId, 'rate', hotelRate)
                self._mipitable.BuildCheckAndSearchTable('rate', hotelRate, hotelId)
                hotelPrice = hotel.find('price').get_text().strip()
                priceList = list()
                self._mipinumber.AddRMBIdentifierToPrice(hotelPrice, priceList)
                self._mipitable.BuildObjectTable(hotelId, 'price', priceList[0])
                self._mipitable.BuildCheckAndSearchTable('price', str(priceList[1]), hotelId)
                hotelAddress = hotel.find('address').get_text().strip()
                self._mipitable.BuildObjectTable(hotelId, 'address', hotelAddress)
    def _HasHotelRequestRule1(self):
        keyword = u'酒店|大酒店|公寓|饭店|标准间|豪华间|双人间|单人间|套间|度假村|hotel|apartment'
        return self._mipiutterance.SearchKeyword(keyword)
    def HasHotelRequest(self):
        if self._HasHotelRequestRule1():
            return True
    def IsQueryHotel(self, mipiUtterance):
        ''' check if hotel query occurrs ... '''
        self._mipiutterance = mipiUtterance
        if not self.HasHotelRequest():
            return False
        nameTypeDict = {'name':0, 'address':0}
        self._mipitable.SetUtterance(self._mipiutterance)
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
            for hotelId in resultDict:
                theHotel = tableDict[hotelId]
                retResult += '{0}\n{1}\n{2}\n{3}\n{4}\n\n'.format(theHotel['name'], theHotel['rate'], theHotel['price'], theHotel['address'])
        else:
            retResult = u'No search results for your query ({0})'.format(self._mipiutterance.GetUtterance())
        return retResult
""" the end of MipiHotel class """
