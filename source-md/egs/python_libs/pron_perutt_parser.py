# 2017 Haihua Xu 
# -*- coding: utf-8 -*-
import re
import os

import sys
# sys.setdefaultencoding('utf8')
import io
import jieba
from operator import itemgetter
import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler)

''' define a class to parse text line
'''
class SimpleTextParser(object):
    def __init__(self, tgtdir, myLogger=None):
        self.pronDict = dict()
        self.tgtdir = tgtdir
        if myLogger:
            self.logger = myLogger
        else:
            self.logger = logger

        if tgtdir:
            try:
                text = '{0}/text'.format(tgtdir)
                self.textFile = open(text, 'w')
            except IOError:
                self.logger.error('Cannot open file {0} for writing !'. format(text))
                sys.exit(1)

    def ParseLine(self, line):
        line = re.sub(r'<s>', '', line)
        line = re.sub(r'</s>', '', line)
        line = re.sub(r'<eps>\s+\S+', '', line)
        # print (line.encode('utf8'))
        line = line.strip('\n')
        wordPronList = line.split('\t')
        self.uttName = wordPronList.pop(0)
        # print('self.uttName is ' + self.uttName)
        # print ('wordPronList is ' + ''.join( wordPronList).encode('utf8'))
        # sys.exit(1)
        utt = self.uttName
        for sPron in wordPronList:
            if(sPron):
                utt += ' ' + sPron.split()[0]
        utt = utt.strip()
        self.logger.debug("Before processing: {0}".format(utt))
        content = ' '. join(wordPronList)
        if(self.IsChineseCharContained(content)):
            self.ResegmentUtterance(wordPronList)
        else:
            self.DumpUtterance(wordPronList)
        # if self.IsSegmentedLine(line):
        #    print ('Line: {0} is already segmented line'. format(line))
        # for index, wordPron in enumerate(wordPronList):
            # if ('<eps>' not in wordPron) and wordPron:
                # print ('after removal: {0}, length: {1}'. format(wordPron, len(wordPron)))
                # if wordPron not in self.pronDict:
                #    self.pronDict[wordPron] = 1
                # else:
                #    self.pronDict[wordPron] += 1
                # if(self.IsChineseWord(wordPron)):
                #    print(wordPron.encode('utf8'))
        # sys.exit(1)
    def GetWord(self, sPron):
        myList = sPron.split()
        return myList[0]
    def GetPhones(self, sPron):
        myList = sPron.split()
        myList.pop(0)
        return ' '.join(myList)
    def CutChineseSegment(self, sWords):
        ''' segment chinese words '''
        unsegWords = sWords;
        myList = jieba.cut(unsegWords)
        sWords = ' '.join(myList)
        return sWords
    def CheckChineseWordPron(self, sWords, sPhones):
        wordList = sWords.split()
        nWord = 0
        for sWord in wordList:
            if(sWord):
                charList = list(sWord)
                nWord += len(charList)
        phoneList = sPhones.split()
        nPhone = len(phoneList)
        if(nPhone == 2*nWord):
            return True
        return False
    def AddResegmentedWordToDict(self, sSegWords, sPhones):
        ''' label resegmented chinese words with pronunciation.
        Here, we use initial and final syllable for each chinese character.
        This means each character has two phones, restrively.
        '''
        if not self.CheckChineseWordPron(sSegWords, sPhones):
            logger.error('word {0} and expected phone number are mismatched (phones are {1})'.format(sSegWords, sPhones))
            sys.exit(1)
        nIndex = 0
        wordList = sSegWords.split()
        phoneList = sPhones.split()
        for sWord in wordList:
            sWordPron = ''
            nChar = len(list(sWord))
            end = nIndex + nChar*2
            start = nIndex
            for x in range(start, end):
                sWordPron += phoneList[x] + ' '
            nIndex = end
            sWordPron = sWord + '\t' + sWordPron;
            sWordPron = sWordPron.strip()
            self.logger.debug('word pronunciation is {}'. format(sWordPron))
            self.AddPronToDict(sWordPron)

    def ResegmentUtterance(self, uttPronList):
        '''use jieba to resegment utterances containing chinese words'''
        sUnsegWords = ''
        sWords = ''
        sPhones = ''
        for sPron in uttPronList:
            if not sPron: continue
            # logger.debug('sPron is {}'. format(sPron))
            sWord = (sPron.split())[0]
            if self.IsChineseWord(sWord):
                sUnsegWords +=  self.GetWord(sPron)
                # logger.debug('sUnsegWords is {}'.format(sUnsegWords))
                sPhones += self.GetPhones(sPron) + ' '
            else:
                if(sUnsegWords):
                    sSegWords = self.CutChineseSegment(sUnsegWords)
                    sWords += sSegWords + ' '
                    self.AddResegmentedWordToDict(sSegWords, sPhones)
                    sUnsegWords = ''
                    sPhones = ''
                sWords += self.GetWord(sPron) + ' '
                self.AddPronToDict(sPron)
        if(sUnsegWords):
            sSegWords = self.CutChineseSegment(sUnsegWords)
            sWords += sSegWords;
            self. AddResegmentedWordToDict(sSegWords, sPhones)
            
        self.logger.debug('After processing: {utt} {words}\n'.format(utt=self.uttName, words=sWords))
        self.textFile.write('{utt} {words}\n'.format(utt=self.uttName, words=sWords))
    def SegmentMixWords(self, sLine):
        sLine = re.sub(r'([^\u4e00-\u9fa5])([\u4e00-\u9fa5])', r'\1 \2', sLine)
        sLine = re.sub(r'([\u4e00-\u9fa5])([^\u4e00-\u9fa5])', r'\1 \2', sLine)
        return sLine
    def DoCharacterSegmentation(self, sLine):
        while(re.search(r'([\u4e00-\u9fa5])([\u4e00-\u9fa5])', sLine)):
            sLine = re.sub(r'([\u4e00-\u9fa5])([\u4e00-\u9fa5])', r'\1 \2', sLine)
        return re.sub(r'\s+', ' ', sLine).strip()
    def SegmentUtterance(self, sLine):
        ''' use jieba to segment utterances '''
        sTokenList = sLine.split()
        sUnsegWords = ''
        sWords = ''
        for sToken in sTokenList:
            if(self.IsChineseWord(sToken)):
                sUnsegWords += sToken
            else:
                if(sUnsegWords):
                    sSegWords = self.CutChineseSegment(sUnsegWords)
                    sWords += sSegWords + ' '
                    sUnsegWords = ''
                sWords += sToken + ' '
        if(sUnsegWords):
            sSegWords = self.CutChineseSegment(sUnsegWords)
            sWords += sSegWords
        return sWords

    def AddPronToDict(self, sPron):
        ''' do basic normalization before insertion'''
        myList = sPron.split()
        sWord = myList.pop(0)
        sPhone = ' '.join(myList)
        sPron = '{word}\t{phones}'.format(word=sWord, phones=sPhone)
        if sPron in self.pronDict:
            self.pronDict[sPron] += 1
        else:
            self.pronDict[sPron] = 1

    def DumpUtterance(self, uttPronList):
        utt = self.uttName
        # logger.debug('utt is: ' + utt)
        # logger.debug('uttPronList is: ' + ' '. join(uttPronList) + ' and type is ' + str(type(uttPronList)) + ' and length is ' + str(len(uttPronList)))
        # logger.debug(uttPronList)
        for sPron in uttPronList:
            if sPron:
                self.AddPronToDict(sPron)
                myList = sPron.split()
                utt += ' ' + myList[0]
        self.logger.debug('After processing: {}'.format(utt))
        self.textFile.write('{0}\n'.format(utt))
            
    def IsChineseCharContained(self, line):
        # self.logger.debug('search results: {0} for {1}'.format(re.search(ur'[\u4e00-\u9fff]+', line.decode('utf8')), type(line)))
        if re.search(r'[\u4e00-\u9fff]+', line):
            return True
        return False
    def IsSegmentedLine(self, line):
        if not self.IsChineseCharContained(line):
            return True
        regex = r'([\u4e00-\u9fff]+)'
        charList = re.findall(regex, line)
        for sWord in charList:
            if(len(list(sWord)) > 1):
                return True
        return False

    def IsChineseWord(self, sWord):
        # sWord = sWord.decode('utf8')
        # self.logger.info('word is {0}, chinese word {1}'.format(sWord, self.IsChineseCharContained(sWord)))
        if(self.IsChineseCharContained(sWord)):
            charList = list(sWord)
            for sChar in charList:
                if(sChar) and self.IsChineseCharContained(sChar): 
                    continue
                else: 
                    self.logger.warning('hybrid word seen {0}'. format(sWord))
                    sys.exit(1)
                    return False
            return True
        return False
    def DumpDictionary(self):
        dictFile = '{tgtdir}/lexicon01.txt'.format(tgtdir=self.tgtdir)
        try:
            with open(dictFile, 'w') as outFile:
                for sPron in self.pronDict:
                    outFile.write('{pron}\t{count}\n'.format(pron=sPron, count=self.pronDict[sPron]))
        except IOError:
            logger.error('cannot open file {0} to write !'. format(dictFile))
            sys.exit(1)
        outFile.close()

class LexiconAnalysor(object):
    def __init__(self, pronCountFile, myLogger = None):
        self.sPronCountFile = pronCountFile
        self.logger = myLogger
        if not self.logger:
            self.logger = logger
        self.parser = SimpleTextParser('', self.logger)
        self.cnDict = dict()
        self.enDict = dict()
        self.mixDict = dict()
        self.nTotalPron = 0
        self.nPrunedPron = 0
        self.tgtdir = ''
    def SetTargetDir(self, tgtdir):
        self.tgtdir = tgtdir
        # self.logger.info("target dir is {}".format(tgtdir))
    def AddWordToDict(self, myDict, sWord, sPhone, nCount):
        sPron = '{word}\t{phone}'.format(word=sWord, phone=sPhone)
        # self.logger.debug('sPron is {0}'.format(sPron))
        sPron = sPron.decode('utf8')
        if sWord in myDict:
            pronDict = myDict[sWord]
            if sPron in pronDict:
                pronDict[sPron] += int(nCount)
            else:
                pronDict[sPron] = int(nCount)
        else:
            pronDict = myDict[sWord] = dict()
            pronDict[sPron] = int(nCount)

    def AddToDict(self, sPron):
        m = re.search(r'^(\S+)\s+(.*)\s+(\d+)$', sPron)        
        if(m):
            sWord = m.group(1).decode('utf8')
            sPhone = m.group(2)
            sPhone = ' '.join(sPhone.split())
            nCount = m.group(3)
            # self.logger.debug('word is {0}'.format(sWord))
            if(self.parser.IsChineseWord(sWord)):
                # self.logger.debug('word {0} is a Chinese word'.format(sWord))
                self.AddWordToDict(self.cnDict, sWord, sPhone, nCount)
            else:
                # self.logger.debug('word {0} is not a Chinese word'. format(sWord))
                self.AddWordToDict(self.enDict, sWord, sPhone, nCount)
        else:
            self.logger.error('Illegal lexicon line: {0}'.format(sPron))
    def LoadLexicon(self):
        with open(self.sPronCountFile, 'r') as lexInputFile:
            for sPron in lexInputFile:
                sPron = sPron.strip()
                if sPron:
                    self.AddToDict(sPron)
        self.logger.info("Loading dictionary {0} successfully !". format(self.sPronCountFile))
        
    def _DumpDict(self, myDict):
        pronList = myDict.items()
        pronList.sort(key=itemgetter(0))
        for sPron in pronList:
            self.logger.debug('Pron: {0}'.format(sPron[0]))
    def GetEnDict(self):
        return self.enDict
    def GetCnDict(self):
        return self.cnDict
    def GetMixDict(self):
        return self.mixDict
    def DumpDict(self):
        self._DumpDict(self.mixDict)
        self.logger.debug('Total pronuciations are {0}, {1} are pruned'.format(self.nTotalPron, self.nPrunedPron))
    def _IsDictOfDict(self, myDict):
        for sWord in myDict:
            valueType = myDict[sWord]
            if(type(valueType) is dict):
                return True
            return False

    def SaveDict(self, dictDict, dictFile):
        with open(dictFile, 'w') as outputFile:
            saveDict = dict()
            if self._IsDictOfDict(dictDict):
                for sWord in dictDict:
                    tmpDict = dictDict[sWord]
                    for sPron in tmpDict:
                        self.AddPronToDict(saveDict, sPron)
            else:
                saveDict = dictDict
            pronList = saveDict.items()
            pronList.sort(key=itemgetter(0))
            for sPron in pronList:
                outputFile.write('{0}\n'.format(sPron[0]))

    def DecomposeWord(self, sPron):
        myList = sPron.split()
        sWord = myList.pop(0).decode('utf8')
        sPhone = ' '.join(myList)
        return [sWord, sPhone]
    def AddPronToDict(self, myDict, sPron):
        if sPron in myDict:
            myDict[sPron] += 1
        else:
            myDict[sPron] = int(1)
    def FilterChineseWords(self):
        self.logger.info("Filtering Chinese word's pronunciation")
        for word in self.cnDict:
            pronDict = self.cnDict[word]
            # self.logger.debug('pronDict size is {0} for word {1}'.format(len(pronDict), word))
            # self._DumpDict(pronDict)
            nSize = len(pronDict)
            self.nTotalPron += nSize
            nChar = len(list(word.decode('utf8')))
            # self.logger.debug('word {0} has {1} characters'.format(word, nChar))
            if nSize > 1 and nChar > 1:
                pronList = pronDict.items()
                self.nPrunedPron += nSize - 1
                pronList.sort(key=itemgetter(1), reverse=True)
                self.AddPronToDict(self.mixDict, pronList[0][0])
            else:
                for sPron in pronDict:
                    self.AddPronToDict(self.mixDict,sPron)
        # self._DumpDict(self.mixDict)
    def FilterEnglishWords(self, pruneThresh = 0.8):
        for word in self.enDict:
            pronDict = self.enDict[word]
            nSize = len(pronDict)
            self.nTotalPron += nSize
            if nSize > 1:
                pronList = pronDict.items()
                pronList.sort(key=itemgetter(1), reverse=True)
                nTotalCount = 0
                for sPron in pronList:
                    nTotalCount += sPron[1]
                nReservedCount = 0
                nReserved = 0
                for sPron in pronList:
                    nReserved += 1
                    nReservedCount += sPron[1]
                    self.AddPronToDict(self.mixDict, sPron[0])
                    if(float(nReservedCount) / nTotalCount > pruneThresh): break
            
                self.nPrunedPron += nSize - nReserved
            else:
                for sPron in pronDict:
                    self.AddPronToDict(self.mixDict,sPron)

            
''' end
'''
