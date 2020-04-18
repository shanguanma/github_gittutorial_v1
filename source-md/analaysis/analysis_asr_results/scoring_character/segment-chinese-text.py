#!/cm/shared/apps/python3.5.2/bin/python3.5

from __future__ import print_function
import re
import os

import sys

import io
import codecs

sys.path.insert(0, 'source-scripts/egs')
import python_libs.pron_perutt_parser as pron_common_lib
import logging
logger = logging.getLogger('libs')
# logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
# handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")

logger.addHandler(handler)

import argparse
parser = argparse.ArgumentParser(description ='segment text containing mandarin words')
parser.add_argument('--debug', dest='debug', help='for user to debug the code')
parser.add_argument('--do-word-segmentation', dest='do_word_segmentation',action='store_true', help='do word segmentation')
parser.add_argument('--do-character-segmentation', dest='do_word_segmentation', action='store_false', help='do character segmentation')
args = parser.parse_args()

if(args.debug):
    logger.setLevel(logging.DEBUG)
    handler.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)
    handler.setLevel(logging.INFO)


myUttParser = pron_common_lib.SimpleTextParser('', logger)
input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8',  errors='ignore')
logger.info("stdin expected")
if args.do_word_segmentation:
    logger.info('do word segmentation')
for  lineno, sLine in enumerate(input_stream, 1):
    logger.debug("{0}".format(sLine))
    sLine = myUttParser.SegmentMixWords(sLine)
    if args.do_word_segmentation:
        sWords = myUttParser.SegmentUtterance(sLine)
    else:
        sWords = myUttParser.DoCharacterSegmentation(sLine)
    print("{0}".format(sWords))
logger.info("stdin ended")
