# -*- coding: utf-8 -*-
"""
Created on Sun Nov 12 03:24:57 2023

@author: farka
"""

import basic_functions as base_fns

print(f'\nTotal line number in texts:'
      f'\nEnglish: {base_fns.line_number_of_text(base_fns.open_text("HP-english_clean.txt"))}'
      f'\nFrench: {base_fns.line_number_of_text(base_fns.open_text("HP-french.txt"))}'
      f'\nGerman: {base_fns.line_number_of_text(base_fns.open_text("HP-german.txt"))}'
      f'\nSpanish: {base_fns.line_number_of_text(base_fns.open_text("HP-spanish.txt"))}')


print(f'\nNumber of words in texts:'
      f'\nEnglish: {base_fns.number_of_words_in_text(base_fns.open_text("HP-english_clean.txt"))}'
      f'\nFrench: {base_fns.number_of_words_in_text(base_fns.open_text("HP-french.txt"))}'
      f'\nGerman: {base_fns.number_of_words_in_text(base_fns.open_text("HP-german.txt"))}'
      f'\nSpanish: {base_fns.number_of_words_in_text(base_fns.open_text("HP-spanish.txt"))}\n')