# -*- coding: utf-8 -*-
"""
Created on Sun Dec  3 15:48:31 2023

@author: adamWolf
"""

import basic_functions as base_fns
import re
import prepare_english as prep_eng
import prepare_german as prep_ger
import prepare_french as prep_fre
import prepare_spanish as prep_esp
import remove_empty_lines as rm_empty_lines
import get_sentences as get_snt


english_text = base_fns.open_text('HP-english.txt').read()
german_text = base_fns.open_text('HP-german.txt').read()
french_text = base_fns.open_text('HP-french.txt').read()
spanish_text = base_fns.open_text('HP-spanish.txt').read()


english_text = prep_eng.prepare_english(english_text)
english_text = rm_empty_lines.remove_empty_lines(english_text)

german_text = prep_ger.prepare_german(german_text)
german_text = rm_empty_lines.remove_empty_lines(german_text)
german_text = re.sub('[»«]','',german_text)

french_text = prep_fre.prepare_french(french_text)
french_text = rm_empty_lines.remove_empty_lines(french_text)

spanish_text = prep_esp.prepare_spanish(spanish_text)
spanish_text = rm_empty_lines.remove_empty_lines(spanish_text)
spanish_text = re.sub('[»«]','',spanish_text)

eng_sentences = get_snt.get_sentences(english_text)
ger_sentences = get_snt.get_sentences(german_text)
fra_sentences = get_snt.get_sentences(french_text)
esp_sentences = get_snt.get_sentences(spanish_text)
print(esp_sentences)


num_eng_sentences = len(eng_sentences)
num_ger_sentences = len(ger_sentences)
num_fra_sentences = len(fra_sentences)
num_esp_sentences = len(esp_sentences)


'''
sentences_set = set(sentences)
sentences_orig_set = set(sentences_orig)

diff = list(sentences_set - sentences_orig_set)
print(diff)
'''

# num_ger = num_snt.num_of_sentences('HP-german_without_pages.txt')
# num_fr = num_snt.num_of_sentences('HP-french.txt')
# num_es = num_snt.num_of_sentences('HP-spanish.txt')

#And \nwhere did you get that motorbike
##This was so unfair that Harry opened his mouth to argue, but \nRon kicked him behind their cauldron.

print(f'number of sentences in english text: {num_eng_sentences}')
print(f'number of sentences in german: {num_ger_sentences}')
print(f'number of sentences in french: {num_fra_sentences}')
print(f'number of sentences in spanish: {num_esp_sentences}')

# print(f'number of sentences in german text: {num_ger}')
# print(f'number of sentences in french text: {num_fr}')
# print(f'number of sentences in spanish text: {num_es}')