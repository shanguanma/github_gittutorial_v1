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

""" define a class to realize text conversion with dictionary
"""
class MipiConvertText(object):
    def __init__(self, mipiDict, mipiNumber, preferLowercase=True, noOovAllowed=True):
        self._mipidict = mipiDict
        self._mipinumber = mipiNumber
        self._lowercase = preferLowercase
        self._nooov = noOovAllowed

    def TransferUtterance(self, sLine):
        lineTransfered = ''
        if not sLine.strip(): return None
        wordList = [x for x in sLine.split() if x.strip()]
        dim = len(wordList)
        if dim == 0: return None
        idx = 0
        while idx < dim:
            word = wordList[idx]
            if self._lowercase: word = word.lower()
            if self._mipinumber.IsTransferable(word):
                word = self._mipinumber.GetTransfer()
                lineTransfered += word + ' '
                idx += 1
                continue
            if idx + 1 < dim:
                nextWord = wordList[idx+1]
                if self._mipinumber.IsTransferable2(word, nextWord):
                    word = self._mipinumber.GetTransfer()
                    lineTransfered += word + ' '
                    idx += 2
                    continue
                if word in self._mipidict.GetWordPronDict():
                    pronDict = self._mipidict.GetPronDict(word)
                    for pron in pronDict:
                        lineTransfered += pron + ' '
                        break
                elif(self._nooov):
                    raise Exception('word {0} is oov word'.format(word))
                else: 
                    lineTransfered += word + ' '
            idx += 1
        lineTransfered = lineTransfered.rstrip()
        return lineTansfered

    def TransferText(self, inTextFile, outTextFile):
        """ transfer text using given dict, some number plus unit will be
            truncated. For instance, 535SGD will be transfered as 540SGD, 
            1230SGD transfered as 1200SGD.
        """
        transfered = str()
        with open(inTextFile, 'r') as istream:
            for line in istream:
                if line.strip():
                    lineTransfered = self.TransferUtterance(line)
                    transfered += lineTransfered + '\n'
        if transfered:
            ostream = open(outTextFile, 'w')
            ostream.write('{0}'.format(transfered))
            ostream.close()

    def LabelUtterance(self, sLine):
        lineLabeled = ''
        if not sLine.strip(): return None
        wordList = [x for x in sLine.split() if x.strip()]
        dim = len(wordList)
        if dim == 0: return None
        idx = 0
        while idx < dim:
            word = wordList[idx]
            if self._lowercase: word = word.lower()
            if self._mipinumber.IsTransferable(word):
                wordLabel = self._mipinumber.LabelNumberAndUnit()
                lineLabeled += wordLabel + ' '
                idx += 1
                continue
            if idx + 1 < dim:
                nextWord = wordList[idx+1]
                if self._lowercase: nextWord = nextWord.lower()
                if self._mipinumber.IsTransferable2(word, nextWord):
                    wordLabel = self._mipinumber.LabelNumberAndUnit()
                    lineLabeled += wordLabel + ' '
                    idx += 2
                    continue
            if word in self._mipidict.GetWordPronDict():
                pronDict = self._mipidict.GetPronDict(word)
                for pron in pronDict:
                    lineLabeled += "{0}\t{1}\n".format(word, pron) + ' '
                    break
            else:
                lineLabeled += "{0}\t{1}\n".format(word, word) + ' '
            idx += 1
        lineLabeled = lineLabeled.rstrip()
        return lineLabeled
        
    def LabelText(self, inTextFile, outTextFile):
        """ label the text with dict
        """
        labeled = str()
        with open(inTextFile, 'r') as istream:
            for line in istream:
                if line.strip():
                    lineLabeled = self.LabelUtterance(line)
                    if labeled: labeled += '\n'
                    labeled += lineLabeled
            if labeled:
                ostream = open(outTextFile, 'w')
                ostream.write('{0}'.format(labeled))
                ostream.close()

    def DumpWordList(self, inTextFile, outTextFile):
        with open(inTextFile, 'r') as istream:
            for line in istream:
                if line.strip():
                    self._mipidict.CountWord(line)
        sortedWordList = self._mipidict.DumpWordList()
        # print ('sortedWordList={0}'.format(sortedWordList))
        words=''
        for word in sortedWordList:
            words +='{0}\n'.format(word)
        ostream = open(outTextFile, 'w')
        ostream.write('{0}'.format(words))
        ostream.close()

""" End of MipiConvertText
"""
