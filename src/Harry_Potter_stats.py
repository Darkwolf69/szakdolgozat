# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 21:50:22 2023
"""

from nltk.corpus import Harry_Potter
from nltk.text import Text

"""
hpEnglishCorpus = Harry_Potter.words("HP-english.txt")
hpEnglishText = Text(hpEnglishCorpus)
hpEnglishText.concordance("Harry")

with open("HP-english.txt", "r", encoding="utf8") as hpEnglish:
    hpEnglishStr = hpEnglish.read()
    hpEnglishWords = hpEnglishStr.words()
    hpEnglishText = Text(hpEnglishWords)
    hpEnglishText.concordance("Harry")
"""


with open("HP-english.txt", "r", encoding="utf8") as hpEnglish:
    hpEnglishTotalLines = len(hpEnglish.readlines())
    print('\nTotal lines of english text:', hpEnglishTotalLines)
    
with open("HP-french.txt", "r", encoding="utf8") as hpFrench:
    hpFrenchTotalLines = len(hpFrench.readlines())
    print('Total lines of french text:', hpFrenchTotalLines)

with open("HP-german.txt", "r", encoding="utf8") as hpGerman:
    hpGermanTotalLines = len(hpGerman.readlines())
    print('Total lines of german text:', hpGermanTotalLines)

with open("HP-spanish.txt", "r", encoding="utf8") as hpSpanish:
    hpSpanishTotalLines = len(hpSpanish.readlines())
    print('Total lines of spanish text:', hpSpanishTotalLines)
    
    
    
englishFile = open("HP-english.txt", "r", encoding="utf8")
number_of_words = 0
data = englishFile.read()
lines = data.split()
number_of_words += len(lines)
print('\nNumber of words in english text:', number_of_words)
englishFile.close()

frenchFile = open("HP-french.txt", "r", encoding="utf8")
number_of_words = 0
data = frenchFile.read()
lines = data.split()
number_of_words += len(lines)
print('Number of words in french text:', number_of_words)
frenchFile.close()

germanFile = open("HP-german.txt", "r", encoding="utf8")
number_of_words = 0
data = germanFile.read()
lines = data.split()
number_of_words += len(lines)
print('Number of words in german text:', number_of_words)
germanFile.close()

spanishFile = open("HP-spanish.txt", "r", encoding="utf8")
number_of_words = 0
data = spanishFile.read()
lines = data.split()
number_of_words += len(lines)
print('Number of words in spanish text:', number_of_words)
spanishFile.close()

"""
with open("HP-english.txt", "r", encoding="utf8") as fp:
    hpEnglishString = fp.read()
    res = list(map(len, hpEnglishString.split()))
    print('\nThe list of words lengths in english text is:\n', res)
"""

    
